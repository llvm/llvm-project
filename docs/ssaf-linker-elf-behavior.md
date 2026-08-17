# ELF Linker Behavior — Grounding Probes

Probes run against `ld.lld` (LLD 24.0.0, built from this tree) to ground the
`LinkageRules` for `Triple::ELF`. Target: `x86_64-unknown-linux-gnu`.

Tools used:

- `CLANG=/Volumes/avirals-external-drive-1/aviralg-llvm-project/bin/clang`
- `LLD=/Volumes/avirals-external-drive-1/aviralg-llvm-project/bin/ld.lld`
- `RO=/Volumes/avirals-external-drive-1/aviralg-llvm-project/bin/llvm-readelf`
- `NM=/Volumes/avirals-external-drive-1/aviralg-llvm-project/bin/llvm-nm`
- `OD=/Volumes/avirals-external-drive-1/aviralg-llvm-project/bin/llvm-objdump`

All objects compiled with:

```
$CLANG --target=x86_64-unknown-linux-gnu -c -O0 <src> -o <obj>
```

---

## 1. Binding precedence and the duplicate-definition error

### Program

`strong_a.c`
```c
int f(void) { return 1; }
```

`strong_b.c`
```c
int f(void) { return 2; }
```

`weak_a.c`
```c
__attribute__((weak)) int f(void) { return 1; }
```

`weak_b.c`
```c
__attribute__((weak)) int f(void) { return 2; }
```

`main.c`
```c
int f(void);
int _start(void) { return f(); }
```

### Commands and output

```
$ $LLD -o out_p1 main.o strong_a.o strong_b.o
ld.lld: error: duplicate symbol: f
>>> defined at strong_a.c
>>>            strong_a.o:(f)
>>> defined at strong_b.c
>>>            strong_b.o:(.text+0x0)
exit=1

$ $LLD -o out_p2 main.o strong_a.o weak_b.o
exit=0

$ $LLD -o out_p3 main.o weak_a.o strong_b.o
exit=0

$ $LLD -o out_p4 main.o weak_a.o weak_b.o
exit=0

$ $LLD -z muldefs -o out_p1b main.o strong_a.o strong_b.o
exit=0
```

Which definition prevailed (`movl $0x1` = `_a`, `movl $0x2` = `_b`):

```
$ $OD -d --disassemble-symbols=f out_p2
00000000002011b0 <f>:
  2011b4: b8 01 00 00 00    movl $0x1, %eax      # strong_a won

$ $OD -d --disassemble-symbols=f out_p3
00000000002011c0 <f>:
  2011c4: b8 02 00 00 00    movl $0x2, %eax      # strong_b won

$ $OD -d --disassemble-symbols=f out_p4
00000000002011b0 <f>:
  2011b4: b8 01 00 00 00    movl $0x1, %eax      # weak_a won (first)

$ $OD -d --disassemble-symbols=f out_p1b
00000000002011b0 <f>:
  2011b4: b8 01 00 00 00    movl $0x1, %eax      # strong_a won (first)
```

### Inference

- **Conflict predicate: two non-weak definitions.** Only `strong+strong`
  errors. This matches `Symbol::checkDuplicate` in `lld/ELF/Symbols.cpp:600`:
  `if (!isWeak() && !other.isWeak()) reportDuplicate(...)`.
- **Strong beats weak regardless of link order.** P2 (strong first) and P3
  (weak first) both keep the strong body. So the winner is decided by binding
  strength, not by position — our `incomingDataWins` must not be
  order-sensitive when strengths differ.
- **Ties keep the first definition.** P4 (weak+weak) and P1b (two strongs under
  `-z muldefs`) both keep the *first* one on the command line. This grounds our
  tie-break rule of "keep the data already linked".
- **`-z muldefs` downgrades the error and keeps the first definition.** This is
  exactly our `WarnOnMultipleDefinitions` flag, and it confirms the semantics
  we chose: on conflict, ignore the incoming occurrence entirely.

---

## 2. Common symbols

### Program

`common_a.c`
```c
int g;
```

`common_big.c`
```c
long long g[4];
```

`def_g.c`
```c
int g = 7;
```

`weakdef_g.c`
```c
__attribute__((weak)) int g = 9;
```

`mainj.c`
```c
extern int g;
int _start(void) { return g; }
```

Compiled with `-fcommon` so tentative definitions become real common symbols.

### Commands and output

Symbol classes in the objects (`C` = common, `D` = data, `V` = weak data):

```
$ $NM common_a.o   | grep " g$"
0000000000000004 C g
$ $NM common_big.o | grep " g$"
0000000000000020 C g
$ $NM def_g.o      | grep " g$"
0000000000000000 D g
$ $NM weakdef_g.o  | grep " g$"
0000000000000000 V g
```

Resolution:

```
$ $LLD -o out_c1 mainj.o common_a.o common_big.o     # common(4) + common(32)
exit=0
$ $RO --symbols out_c1 | grep " g$"
     5: 0000000000203220    32 OBJECT  GLOBAL DEFAULT     5 g

$ $LLD -o out_c2 mainj.o common_a.o def_g.o          # common(4) + strong data(4)
exit=0
$ $RO --symbols out_c2 | grep " g$"
     5: 0000000000203220     4 OBJECT  GLOBAL DEFAULT     5 g

$ $LLD -o out_c2b mainj.o common_big.o def_g.o       # common(32) + strong data(4)
exit=0
$ $RO --symbols out_c2b | grep " g$"
     5: 0000000000203220     4 OBJECT  GLOBAL DEFAULT     5 g

$ $LLD -o out_c3 mainj.o weakdef_g.o common_big.o    # weak data(4) + common(32)
exit=0
$ $RO --symbols out_c3 | grep " g$"
     5: 0000000000203230    32 OBJECT  GLOBAL DEFAULT     6 g
```

### Inference

- **common + common merges with `size = max`** (C1: 4 and 32 give 32). Matches
  `Symbol::resolve(const CommonSymbol &)` at `lld/ELF/Symbols.cpp:618-626`.
- **A strong definition beats common regardless of size** (C2b: common of 32
  loses to strong data of 4, final size 4). So this is a strength comparison,
  not a size comparison — `Strong > Common`.
- **Common beats a weak definition** (C3: final size 32, the common's). This is
  the `isDefined() && !isWeak()` early-return at `:612` — a *weak* definition
  does not stop the common from overwriting it. So `Common > Weak`.
- No case here errors: two commons are legal (only `--warn-common` mentions
  them). Our conflict predicate must not fire on commons.

Net ELF strength order: **`Undefined < Weak < Common < Strong`**, which is the
order our `EntityBinding` enum should encode.

---

## 3. Inline functions: how ODR is represented

### Program

`inl.h`
```c
inline int inl(void) { return 42; }
```

`inl_a.cpp`
```c
#include "inl.h"
int use_a(void) { return inl(); }
```

`inl_b.cpp`
```c
#include "inl.h"
int use_b(void) { return inl(); }
```

### Commands and output

```
$ $NM inl_a.o | grep -i inl
0000000000000000 W _Z3inlv

$ $RO --symbols inl_a.o | grep "_Z3inlv"
     3: 0000000000000000     0 SECTION LOCAL  DEFAULT     5 .text._Z3inlv
     5: 0000000000000000    11 FUNC    WEAK   DEFAULT     5 _Z3inlv

$ $RO --section-groups inl_a.o
COMDAT group section [    4] `.group' [_Z3inlv] contains 1 sections:
   [Index]    Name
   [    5]   .text._Z3inlv
```

### Inference

**There is no `WeakODR` binding on ELF.** An `inline` function is emitted as
`STB_WEAK` (binding) *plus* a COMDAT group (coalescing). The two properties are
carried by separate mechanisms in the object file.

This is the direct justification for splitting `EntityBinding` and
`EntityCoalescing` into independent fields: an ODR definition and a plain
`__attribute__((weak))` definition have *identical* binding, and differ only in
whether the copies are guaranteed identical. Ranking `WeakODR` above `Weak`, as
a single enum forces, models a precedence the linker does not have.

---

## 4. Visibility merging

### Program

`vis_def.c`
```c
int v(void) { return 1; }
```

`vis_hidden_decl.c`
```c
__attribute__((visibility("hidden"))) int v(void);
int caller(void) { return v(); }
```

`vis_prot_decl.c`
```c
__attribute__((visibility("protected"))) int v(void);
int caller2(void) { return v(); }
```

Compiled with `-fPIC`; linked with `-shared` so the export table is meaningful.

### Commands and output

```
$ $LLD -shared -o out_v2.so vis_def.o                      # control: default only
$ $RO --dyn-syms out_v2.so | grep " v$"
     1: 0000000000001280    11 FUNC    GLOBAL DEFAULT     6 v

$ $LLD -shared -o out_v1.so vis_def.o vis_hidden_decl.o    # default def + hidden decl
exit=0
$ $RO --dyn-syms out_v1.so | grep " v$"
   (v NOT exported)
$ $RO --symbols out_v1.so | grep " v$"
     4: 00000000000012a0    11 FUNC    LOCAL  HIDDEN      6 v

$ $LLD -shared -o out_v3.so vis_def.o vis_prot_decl.o      # default def + protected decl
exit=0
$ $RO --dyn-syms out_v3.so | grep " v$"
     1: 00000000000012c0    11 FUNC    GLOBAL PROTECTED   6 v
```

### Inference

- **Visibility merges to the most restrictive across *all* occurrences,
  including ones that lose symbol resolution.** In V1 the definition has
  default visibility and only an *undefined declaration* is hidden, yet the
  merged symbol becomes `LOCAL HIDDEN`. This matches `Symbol::mergeProperties`
  (`:415`) and the visibility handling at the top of each `resolve()` overload:
  `setVisibility(v == STV_DEFAULT ? ov : std::min(v, ov))` over
  `STV_DEFAULT=0 < STV_INTERNAL=1 < STV_HIDDEN=2 < STV_PROTECTED=3`.

  Note the ELF encoding orders `PROTECTED` *above* `HIDDEN` numerically but
  `min` is applied, and `DEFAULT` is special-cased — so the effective
  restrictiveness order is `DEFAULT < PROTECTED < HIDDEN`, matching our
  `EntityVisibility` numbering with `std::max`.

- **This is the grounding for "Hidden blocks cross-LU resolution".** A hidden
  symbol is removed from the dynamic symbol table entirely: it resolves within
  the link unit and is invisible past its boundary. Our LU analogue is that a
  `Hidden` entity must not be resolvable by a later link stage.

- **`Protected` is exported but distinct from `Default`.** V3 keeps `v` in the
  dynamic table with `PROTECTED`, meaning callers inside the DSO bind directly
  and it cannot be interposed. This is a real, ELF-only third state — it is not
  representable on Mach-O or COFF.

---

## 5. Undefined symbols

### Program

`undef_main.c`
```c
int missing(void);
int _start(void) { return missing(); }
```

`undef_weak_main.c`
```c
__attribute__((weak)) int maybe_missing(void);
int _start(void) { return maybe_missing ? maybe_missing() : 0; }
```

### Commands and output

```
$ $LLD -o out_u1 undef_main.o                                  # strong ref, executable
ld.lld: error: undefined symbol: missing
>>> referenced by undef_main.c
>>>               undef_main.o:(_start)
exit=1

$ $LLD -o out_u2 undef_weak_main.o                             # weak ref, executable
exit=0

$ $LLD -shared -o out_u3.so undef_main.o                       # strong ref, shared lib
exit=0

$ $LLD --unresolved-symbols=ignore-all -o out_u4 undef_main.o
exit=0

$ $LLD --warn-unresolved-symbols -o out_u5 undef_main.o
ld.lld: warning: undefined symbol: missing
>>> referenced by undef_main.c
>>>               undef_main.o:(_start)
exit=0
```

### Inference

- **An unresolved strong reference is an error when linking an executable**
  (U1), but **not** when linking a shared library (U3), where it is expected to
  be satisfied at load time.
- **An unresolved *weak* reference is never an error** (U2) — it resolves to
  address 0 and the program tests for it.
- The policy is configurable three ways (`ReportError`/`Warn`/`Ignore`, see
  `UnresolvedPolicy` in `lld/ELF/Config.h:115`), and ELF keeps **two separate
  policies**: `unresolvedSymbols` and `unresolvedSymbolsInShlib` (`:468-469`).

For our linker this means undefined-symbol reporting needs:

1. A tri-state policy, not a bool — matching `--unresolved-symbols`.
2. Suppression for entities whose merged binding is `Weak` (an undefined weak
   reference is legal by design).
3. Awareness that "is this LU a final executable or an intermediate library?"
   changes the answer. An LU that is later combined with others is the
   shared-library case, where unresolved references are normal.

Point 3 is significant for SSAF: a single LU is almost always an intermediate
artifact, so the default should be permissive, with the strict check applied
only when the link unit is known to be complete.

---

## 6. Second pass: ordering details

These probes were added to pin down the exact rank tables for `LinkageRules`,
rather than inferring them from the LLD source.

### 6.1 Weak declaration + common definition

The binding join must not simply take the strongest binding across *all*
occurrences, because a declaration's binding describes how the reference binds,
not the definition. This probe isolates that case.

`weakdecl.c`
```c
__attribute__((weak)) extern int g;
int use(void) { return g; }
```

`commondef.c`
```c
int g;
```

`rd.c`
```c
extern int g; int use(void);
int MAINSYM(void) { return g + use(); }
```

Compiled with `-fcommon -DMAINSYM=_start`.

```
$ $NM e_weakdecl.o  | grep " g$"
                 w g                      # lowercase w = weak UNDEFINED
$ $NM e_commondef.o | grep " g$"
0000000000000004 C g

$ $LLD -o e_out e_rd.o e_weakdecl.o e_commondef.o
exit=0
$ $RO --symbols e_out | grep " g$"
     5: 0000000000203270     4 OBJECT  GLOBAL DEFAULT     5 g
```

**Inference.** The merged symbol is `GLOBAL`, not `WEAK`: the weak *declaration*
contributed nothing to the binding. The definition alone determined it.

This confirms the binding join must be scoped to defining occurrences:

```
if any occurrence is a Definition:  join over the defining occurrences only
else:                               join over the declarations
```

Without the gate, a Mach-O-style ordering (where `Weak` outranks `Common`)
would produce `Weak` here, wrongly describing a common definition as weak.

### 6.2 Protected vs Hidden restrictiveness

Section 4 established `Default` is least restrictive, but compared each of
`Hidden` and `Protected` only against `Default`. This probe compares them
directly, in both orders.

`vp_def_prot.c`
```c
__attribute__((visibility("protected"))) int v(void) { return 1; }
```

`vp_decl_hidden.c`
```c
__attribute__((visibility("hidden"))) int v(void);
int c1(void) { return v(); }
```

`vp_def_hidden.c`
```c
__attribute__((visibility("hidden"))) int v(void) { return 1; }
```

`vp_decl_prot.c`
```c
__attribute__((visibility("protected"))) int v(void);
int c2(void) { return v(); }
```

```
$ $LLD -shared -o o_vp1.so vp_def_prot.o vp_decl_hidden.o     # protected def + hidden decl
$ $RO --symbols o_vp1.so | grep " v$"
     4: 00000000000012a0    11 FUNC    LOCAL  HIDDEN      6 v

$ $LLD -shared -o o_vp2.so vp_def_hidden.o vp_decl_prot.o     # hidden def + protected decl
$ $RO --symbols o_vp2.so | grep " v$"
     4: 00000000000012a0    11 FUNC    LOCAL  HIDDEN      6 v
```

**Inference.** `Hidden` wins over `Protected` in both orders, so the merge is
commutative and the ELF restrictiveness order is:

    Default < Protected < Hidden

Note this is *not* the numeric order of the ELF constants
(`STV_DEFAULT=0, STV_INTERNAL=1, STV_HIDDEN=2, STV_PROTECTED=3`) — LLD applies
`min` with `STV_DEFAULT` special-cased, which yields the order above. The rank
table must encode the observed behaviour, not the ELF constant values.

### ELF rank tables

Derived from sections 1, 2, 4 and 6:

| `EntityBinding` | rank | evidence |
|---|---|---|
| `Undefined` | 0 | any definition beats an undefined reference (U1/U2) |
| `Weak` | 1 | common beats weak def (C3) |
| `Common` | 2 | strong beats common (C2b), common beats weak (C3) |
| `Strong` | 3 | duplicate error only for strong+strong (P1) |

| `EntityVisibility` | rank | evidence |
|---|---|---|
| `Default` | 0 | V1: hidden decl demotes default def |
| `Protected` | 1 | VP1/VP2: loses to hidden |
| `Hidden` | 2 | VP1/VP2: wins both orders |

Conflict predicate: **both occurrences are definitions and neither is `Weak`**
(P1 errors; P2/P3/P4 do not; commons never conflict).

---

## 7. Third pass: visibility on common symbols

Section 4 established the visibility merge using function definitions. This
probe confirms it applies to commons too, which are handled by a separate
`resolve()` overload in LLD.

`hc_hidden_common.c`
```c
__attribute__((visibility("hidden"))) int g;
```

`hc_default_common.c`
```c
int g;
```

Compiled with `-fcommon -fPIC`.

```
$ $NM e_hc.o | grep " g$"
0000000000000004 C g

$ $LLD -shared -o e_hc.so e_hc.o e_dc.o
$ $RO --symbols e_hc.so | grep " g$"
     4: 00000000000032b8     4 OBJECT  LOCAL  HIDDEN      8 g
$ $RO --dyn-syms e_hc.so | grep " g$"
   (NOT exported)
```

### Inference

A hidden common merged with a default-visibility common yields `LOCAL HIDDEN`
and is not exported — the same most-restrictive rule as for function
definitions. This matches `Symbol::resolve(const CommonSymbol &)` at
`lld/ELF/Symbols.cpp:607-611`, which applies the identical
`setVisibility(v == STV_DEFAULT ? ov : std::min(v, ov))` step before any
common-specific handling.

So on ELF the visibility rule is uniform across all symbol kinds, and our
single `visibilityRank` table is sufficient. (Mach-O is not uniform — see §7.2
of the Mach-O doc.)

---

## 8. Fourth pass: ODR definition vs regular definition

This is the case that decides whether the extractor can be target-independent.
COFF errors here (COFF §2, probe K4); this probe establishes what ELF does.

### Program

`inl.h`
```c
inline int q(void) { return 42; }
```

`e_inl.cpp`
```c
#include "inl.h"
int use_inl(void) { return q(); }
```

`e_inl_b.cpp`
```c
#include "inl.h"
int use_inl_b(void) { return q(); }
```

`e_regular.cpp`
```c
int q(void) { return 99; }
int use_reg(void) { return 0; }
```

`e_main.cpp`
```c
int use_inl(void); int use_reg(void);
extern "C" int _start(void) { return use_inl() + use_reg(); }
```

### Commands and output

```
$ $NM e_inl.o     | grep _Z1qv
0000000000000000 W _Z1qv                  # weak + COMDAT
$ $NM e_regular.o | grep _Z1qv
0000000000000000 T _Z1qv                  # strong

$ $LLD -o o_ei1 e_main.o e_inl.o e_regular.o        # ODR + regular strong
exit=0
$ $OD -d --disassemble-symbols=_Z1qv o_ei1
  201234: b8 63 00 00 00    movl $0x63, %eax        # 99 — the regular def won

$ $LLD -o o_ei2 e_main2.o e_inl.o e_inl_b.o         # ODR + ODR
exit=0
```

### Inference

**ELF does not error on an ODR definition colliding with a regular
definition** — the regular (strong) definition simply wins, because the inline
was lowered to `STB_WEAK`. Two ODR definitions also link cleanly.

Contrast COFF probe K4, where the identical source is a `duplicate symbol`
error, because COFF keeps the inline definition `Strong` and relies on COMDAT
to license duplicates.

The same probe on Mach-O (`arm64-apple-macosx`, same sources) also links
cleanly with the regular definition winning:

```
$ ld -o o_mi1 m_main.o m_inl.o m_regular.o ...
ld exit=0
$ $OD -d --disassemble-symbols=__Z1qv o_mi1
100000370: 52800c60    mov w0, #0x63    ; =99
```

### Consequence for the model

The three platforms disagree on this case *only because they lower `inline`
differently*, not because they disagree about C++ semantics. The summary should
therefore record the source-level fact — `Strong` + `ODR`, which is what COFF
encodes directly — and let `LinkageRules` perform the platform lowering:

    effectiveBinding(Strong, ODR) = Weak     on ELF and Mach-O
    effectiveBinding(Strong, ODR) = Strong   on COFF

With that, one conflict predicate per platform reproduces all four observed
rows without the extractor needing to know the target:

| case | effective bindings | predicate | observed |
|---|---|---|---|
| ELF: ODR + regular | `Weak` vs `Strong` | no conflict | EI1 ok |
| ELF: ODR + ODR | `Weak` vs `Weak` | no conflict | EI2 ok |
| COFF: ODR + regular | `Strong` vs `Strong`, not both ODR | conflict | K4 error |
| COFF: ODR + ODR | `Strong` vs `Strong`, both ODR | no conflict | CI1 ok |

---

## 9. Fifth pass: two declarations with differing bindings

The only join case left unprobed: a strong and a weak *declaration* of the same
entity, with no definition anywhere in the link.

### Program

`strongdecl.c`
```c
extern int g;
int use_strong_decl(void) { return g; }
```

`weakdecl.c`
```c
__attribute__((weak)) extern int g;
int use_weak_decl(void) { return g ? 1 : 0; }
```

`dmain.c`
```c
int use_strong_decl(void); int use_weak_decl(void);
int MAINSYM(void) { return use_strong_decl() + use_weak_decl(); }
```

`wonly_main.c`
```c
int use_weak_decl(void);
int _start(void) { return use_weak_decl(); }
```

Compiled with `-DMAINSYM=_start`.

### Commands and output

```
$ $NM e_sd.o | grep " g$"
                 U g                       # uppercase U = undefined strong
$ $NM e_wd.o | grep " g$"
                 w g                       # lowercase w = undefined WEAK

$ $LLD -o e_d1 e_dm.o e_wd.o e_sd.o        # weak decl FIRST
ld.lld: error: undefined symbol: g
>>> referenced by strongdecl.c
>>>               e_sd.o:(use_strong_decl)
>>> referenced by weakdecl.c

$ $LLD -o e_d2 e_dm.o e_sd.o e_wd.o        # strong decl FIRST
ld.lld: error: undefined symbol: g
   (identical diagnostic)
```

Control — the weak declaration alone is not an error:

```
$ $LLD -o e_d3 e_wm.o e_wd.o
exit=0
$ $RO --symbols e_d3 | grep " g$"
     5: 0000000000000000     0 NOTYPE  WEAK   DEFAULT   UND g
```

The merged binding, read from a shared library where unresolved references are
permitted:

```
$ $LLD -shared -o e_d4.so e_dm.o e_sd.o e_wd.o
$ $RO --dyn-syms e_d4.so | grep " g$"
     1: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT   UND g
```

### Inference

**Strong wins over weak among declarations, in either order.** D4 shows the
merged undefined symbol is `GLOBAL`, not `WEAK`, and D1/D2 show the resulting
error is order-independent. The control (D3) confirms a weak-only reference is
legal, so the error in D1/D2 is caused by the strong reference dominating.

This validates the plan's assumption that two declarations join by
`argmax strengthRank`, with `Undefined < Weak` on every platform.

It also settles the one place the result is observable: Phase 3's
undefined-symbol reporting suppresses the diagnostic when the merged binding is
`Weak`. Because a strong reference dominates, an entity referenced weakly by
one TU and strongly by another is correctly reported — matching D1/D2.




