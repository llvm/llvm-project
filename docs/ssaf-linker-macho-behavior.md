# Mach-O Linker Behavior — Grounding Probes

Probes run against Apple's system linker (`ld-1221.4`, i.e. ld-prime — the real
production linker, not LLD) to ground the `LinkageRules` for `Triple::MachO`.
Target: `arm64-apple-macosx`.

Tools used:

- `CLANG=/Volumes/avirals-external-drive-1/aviralg-llvm-project/bin/clang`
- `LD=/usr/bin/ld` (`ld -v` → `PROGRAM:ld PROJECT:ld-1221.4`)
- `NM=/Volumes/avirals-external-drive-1/aviralg-llvm-project/bin/llvm-nm`
- `RO=/Volumes/avirals-external-drive-1/aviralg-llvm-project/bin/llvm-readobj`
- `OD=/Volumes/avirals-external-drive-1/aviralg-llvm-project/bin/llvm-objdump`
- `SDK=$(xcrun --show-sdk-path)`

All objects compiled with:

```
$CLANG --target=arm64-apple-macosx -c -O0 <src> -o <obj>
```

All links use the helper:

```sh
run() {
  ld -o "$1" "${@:2}" -syslibroot "$SDK" -lSystem -arch arm64 2>/tmp/err.txt
  local rc=$?
  grep -v "newer 'macOS'" /tmp/err.txt   # filter SDK-version noise
  echo "ld exit=$rc"
}
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
int main(void) { return f(); }
```

### Commands and output

How weakness is encoded in the object file:

```
$ $RO --syms strong_a.o | grep -A6 "Name: _f"
    Name: _f (1)
    Extern
    Type: Section (0xE)
    Section: __text (0x1)
    RefType: UndefinedNonLazy (0x0)
    Flags [ (0x0)
    ]

$ $RO --syms weak_a.o | grep -A6 "Name: _f"
    Name: _f (1)
    Extern
    Type: Section (0xE)
    Section: __text (0x1)
    RefType: UndefinedNonLazy (0x0)
    Flags [ (0x80)
      WeakDef (0x80)
```

Resolution:

```
$ ld -o out_m1 main.o strong_a.o strong_b.o ...
duplicate symbol '_f' in:
    /private/tmp/ssaf-probe/macho/strong_b.o
    /private/tmp/ssaf-probe/macho/strong_a.o
ld: 1 duplicate symbols
exit=1

$ ld -o out_m2 main.o strong_a.o weak_b.o ...      # strong + weak
exit=0
$ ld -o out_m3 main.o weak_a.o strong_b.o ...      # weak + strong
exit=0
$ ld -o out_m4 main.o weak_a.o weak_b.o ...        # weak + weak
exit=0
```

Which definition prevailed (`mov w0, #0x1` = `_a`, `#0x2` = `_b`):

```
$ $OD -d --disassemble-symbols=_f out_m2
100000348: 52800020    mov w0, #0x1     # strong_a won
$ $OD -d --disassemble-symbols=_f out_m3
100000348: 52800040    mov w0, #0x2     # strong_b won
$ $OD -d --disassemble-symbols=_f out_m4
100000430: 52800020    mov w0, #0x1     # weak_a won (first)
```

### Inference

Identical to ELF on all three counts:

- **Conflict predicate: two non-weak definitions.** Matches
  `SymbolTable::addDefined` in `lld/MachO/SymbolTable.cpp:113-156`, which
  pushes a `dupSymDiags` entry only in the `else` branch where neither the
  existing nor the incoming definition is weak.
- **Strong beats weak regardless of order** (M2 and M3 both keep the strong
  body).
- **Ties keep the first definition** (M4).

Note: the ld64 `-m` flag (allow multiple definitions) is obsolete in ld-prime —
`ld: warning: -m is obsolete` and the duplicate error still fires. So on Mach-O
there is no supported way to downgrade the duplicate-symbol error, unlike ELF's
`-z muldefs`. Our `WarnOnMultipleDefinitions` flag is therefore an
SSAF-specific affordance on this platform, not an emulation of a linker option.

---

## 2. Visibility: `privateExtern`, and the divergence from ELF

### Program

`pe_hidden.c`
```c
__attribute__((visibility("hidden"))) int v(void) { return 1; }
```

`pe_default.c`
```c
int v(void) { return 1; }
```

`pe_prot.c`
```c
__attribute__((visibility("protected"))) int v(void) { return 1; }
```

Two weak definitions differing in visibility:

`wd_hidden.c`
```c
__attribute__((weak, visibility("hidden"))) int w(void) { return 1; }
int anchor_a(void) { return 0; }
```

`wd_hidden2.c`
```c
__attribute__((weak, visibility("hidden"))) int w(void) { return 2; }
int anchor_b(void) { return 0; }
```

`wd_default.c`
```c
__attribute__((weak)) int w(void) { return 2; }
int anchor_b(void) { return 0; }
```

`wd_main.c`
```c
int w(void);
int anchor_a(void); int anchor_b(void);
int main(void) { return w() + anchor_a() + anchor_b(); }
```

### Commands and output

`protected` is rejected outright at compile time:

```
$ $CLANG --target=arm64-apple-macosx -c -O0 pe_prot.c -o pe_prot.o
pe_prot.c:1:16: warning: target does not support 'protected' visibility;
                         using 'default' [-Wunsupported-visibility]
    1 | __attribute__((visibility("protected"))) int v(void) { return 1; }
      |                ^
```

`hidden` becomes the `PrivateExtern` bit, not a visibility field:

```
$ $RO --syms pe_hidden.o | grep -A8 "Name: _v"
    Name: _v (1)
    PrivateExtern
    Extern
    ...

$ $RO --syms pe_default.o | grep -A8 "Name: _v"
    Name: _v (1)
    Extern
    ...
```

Coalescing two weak definitions with differing visibility:

```
$ ld -o out_w1 wd_main.o wd_hidden.o wd_default.o ...   # hidden first
$ $NM -m out_w1 | grep " _w$"
000000010000045c (__TEXT,__text) weak external _w

$ ld -o out_w2 wd_main.o wd_default.o wd_hidden.o ...   # default first
$ $NM -m out_w2 | grep " _w$"
0000000100000454 (__TEXT,__text) weak external _w

$ ld -o out_w3 wd_main.o wd_hidden.o wd_hidden2.o ...   # BOTH hidden
$ $NM -m out_w3 | grep " _w$"
000000010000036c (__TEXT,__text) non-external (was a private external) _w
```

### Inference

- **Mach-O has no visibility enum — only a `PrivateExtern` bit.** There are two
  states, not three. `Protected` is not merely unused on this target: clang
  refuses to emit it and silently downgrades to `default`.

- **Visibility merges to the LEAST restrictive, the opposite of ELF.** W1 and
  W2 (one hidden, one default, either order) both produce an *exported* symbol;
  only W3 (both hidden) stays private. This is
  `defined->privateExtern &= isPrivateExtern` at
  `lld/MachO/SymbolTable.cpp:118` — private only if *every* copy is private.

  Compare ELF probe V1, where a hidden *declaration* was enough to make a
  default-visibility definition `LOCAL HIDDEN`. The two platforms take opposite
  directions, so this rule cannot be shared.

- Order does not matter (W1 == W2), so the rule is still commutative — just
  `min`-of-restrictiveness rather than `max`.

---

## 3. Common symbols

### Program

`common_a.c` → `int g;`, `common_big.c` → `long long g[4];`,
`def_g.c` → `int g = 7;`, `weakdef_g.c` → `__attribute__((weak)) int g = 9;`,
`mainj.c` → `extern int g; int main(void) { return g; }`

Compiled with `-fcommon`.

### Commands and output

```
$ $NM common_a.o   | grep " _g$"
0000000000000004 C _g
$ $NM common_big.o | grep " _g$"
0000000000000020 C _g
$ $NM def_g.o      | grep " _g$"
0000000000000000 D _g
$ $NM weakdef_g.o  | grep " _g$"
0000000000000000 D _g
```

```
$ run out_mc1 mainj.o common_a.o common_big.o        # common(4) + common(32)
ld exit=0
$ $NM -m out_mc1 | grep " _g$"
0000000100004000 (__DATA,__common) external _g

$ run out_mc2 mainj.o common_a.o def_g.o             # common(4) + strong data(4)
ld exit=0
$ $NM -m out_mc2 | grep " _g$"
0000000100004000 (__DATA,__data) external _g

$ run out_mc2b mainj.o common_big.o def_g.o          # common(32) + strong data(4)
ld exit=0
$ $NM -m out_mc2b | grep " _g$"
0000000100004000 (__DATA,__data) external _g

$ run out_mc3 mainj.o weakdef_g.o common_big.o       # weak data(4) + common(32)
ld: warning: tentative definition of '_g' with size 32 from
    '.../common_big.o' is being replaced by real definition of smaller
    size 4 from '.../weakdef_g.o'
ld exit=0
$ $NM -m out_mc3 | grep " _g$"
0000000100008000 (__DATA,__data) weak external _g
```

### Inference

- **Two commons merge, no error** (MC1), as on ELF.
- **A strong definition beats common regardless of size** (MC2b), as on ELF.
- **A WEAK definition also beats common** (MC3) — and Mach-O *warns* about the
  size shrink. **This is a divergence from ELF**, where the common won over a
  weak definition (ELF probe C3). It matches `lld/MachO/SymbolTable.cpp:259`,
  where `addCommon` returns early for any `isa<Defined>(s)` without consulting
  weakness, versus ELF's `isDefined() && !isWeak()` guard at
  `lld/ELF/Symbols.cpp:612`.

So the strength order differs between platforms:

- ELF: `Undefined < Weak < Common < Strong`
- Mach-O: `Undefined < Common < Weak < Strong`

A single global `EntityBinding` ordering cannot express both. The comparison
must live in `LinkageRules`, not in the enum's numeric values.

---

## 4. Inline functions: how ODR is represented

### Program

`inl.h`
```c
inline int inl(void) { return 42; }
```

`inl_a.cpp` / `inl_b.cpp` — each `#include "inl.h"` and call `inl()`.

### Commands and output

```
$ $NM -m inl_a.o | grep -i inlv
0000000000000014 (__TEXT,__text) weak external __Z3inlv

$ $RO --syms inl_a.o | grep -A8 "Name: __Z3inlv"
    Name: __Z3inlv (1)
    Extern
    Type: Section (0xE)
    Section: __text (0x1)
    RefType: UndefinedNonLazy (0x0)
    Flags [ (0x80)
      WeakDef (0x80)
    ]
```

### Inference

**On Mach-O an ODR definition and a plain `__attribute__((weak))` definition
are byte-for-byte identical in the symbol table** — both are just `WeakDef`
(0x80). Compare section 1, where `weak_a.o`'s `_f` carried exactly the same
flag.

Unlike ELF, there is not even a COMDAT group to distinguish them: the ODR
guarantee is entirely absent from the object file. This reinforces that
`Coalescing` is *not* recoverable from a linker's view of Mach-O — it is
front-end knowledge that SSAF must carry explicitly if it wants it, which is
precisely the argument for a separate `EntityCoalescing` field sourced from
clang's `GVA_*ODR` linkage rather than inferred from the binding.

---

## 5. Undefined symbols

### Program

`undef_main.c`
```c
int missing(void);
int main(void) { return missing(); }
```

`undef_weak_main.c`
```c
__attribute__((weak_import)) int maybe_missing(void);
int main(void) { return maybe_missing ? maybe_missing() : 0; }
```

### Commands and output

```
$ run out_mu1 undef_main.o                              # strong ref, executable
Undefined symbols for architecture arm64:
  "_missing", referenced from:
      _main in undef_main.o
ld: symbol(s) not found for architecture arm64
ld exit=1

$ run out_mu2 undef_weak_main.o                         # weak_import ref, executable
Undefined symbols for architecture arm64:
  "_maybe_missing", referenced from:
      _main in undef_weak_main.o
ld: symbol(s) not found for architecture arm64
ld exit=1

$ ld -dylib -o out_mu3.dylib undef_main.o ...           # strong ref, DYLIB
ld exit=1
Undefined symbols for architecture arm64:
  "_missing", referenced from:
      _main in undef_main.o

$ ld -o out_mu4 undef_main.o -undefined dynamic_lookup ...
ld exit=0
```

### Inference

Mach-O is **stricter than ELF** in two ways that matter to us:

- **A dylib with unresolved references is an error** (MU3), whereas ELF happily
  produced `out_u3.so` (ELF probe U3). Mach-O requires every symbol to be
  resolved at static-link time unless told otherwise; ELF defers to load time.
- **`weak_import` does not by itself excuse an entirely absent symbol** (MU2).
  It marks a symbol that may be missing *at runtime* when it comes from a
  dylib; with no definition anywhere in the link, it is still an error.

`-undefined dynamic_lookup` (MU4) is the escape hatch, analogous to ELF's
`--unresolved-symbols=ignore-all`.

For our linker: the "is this LU final or intermediate?" distinction that ELF
makes via shared-vs-executable does **not** hold on Mach-O. If we want one
policy across platforms, an explicit flag is the honest way to express it,
rather than inferring permissiveness from the artifact kind.

---

## 6. Second pass: ordering details

### 6.1 Weak declaration + common definition

Same program as the ELF doc section 6.1 (`weakdecl.c`, `commondef.c`, `rd.c`),
compiled with `-fcommon -DMAINSYM=main`.

```
$ $NM m_weakdecl.o  | grep " _g$"
                 U _g                     # plain undefined
$ $NM m_commondef.o | grep " _g$"
0000000000000004 C _g

$ ld -o m_out m_rd.o m_weakdecl.o m_commondef.o ...
ld exit=0
$ $NM -m m_out | grep " _g$"
0000000100004000 (__DATA,__common) external _g
```

**Inference.** The merged symbol is the common definition, plain `external`,
with no trace of the weak declaration. Same conclusion as ELF: the binding join
is scoped to defining occurrences.

This case matters more on Mach-O than on ELF, because Mach-O ranks `Weak` above
`Common` (section 3). Without the definition-kind gate, joining a weak
declaration with a common definition would yield `Weak`, which the linker
demonstrably does not do.

Note also that clang did not even mark the reference weak here — `nm` shows a
plain `U`, not a weak undefined — so on Mach-O `__attribute__((weak))` on a
declaration is largely inert.

### Mach-O rank tables

Derived from sections 1, 2, 3 and 6:

| `EntityBinding` | rank | evidence |
|---|---|---|
| `Undefined` | 0 | any definition beats an undefined reference (MU1) |
| `Common` | 1 | weak def replaces common, with a warning (MC3) |
| `Weak` | 2 | strong beats weak (M2/M3); weak beats common (MC3) |
| `Strong` | 3 | duplicate error only for strong+strong (M1) |

**This inverts `Weak` and `Common` relative to ELF and COFF** — the single
strongest argument for moving the ordering out of the enum and into
`LinkageRules`.

| `EntityVisibility` | rank | evidence |
|---|---|---|
| `Default` | 1 | W1/W2: hidden + default coalesce to *exported* |
| `Hidden` | 0 | W3: private only when every copy is private |
| `Protected` | — | not representable; clang downgrades to `Default` |

**The visibility ranks are inverted relative to ELF**: here the *least*
restrictive wins, so `Default` outranks `Hidden`. Expressing both platforms
with one "take the greater rank" join is exactly why the ranks must be supplied
per-target rather than baked into the enum's numeric values.

Conflict predicate: **both occurrences are definitions and neither is `Weak`**
(M1 errors; M2/M3/M4 do not) — the same predicate as ELF, unlike COFF.

---

## 7. Third pass: coalescing and visibility-on-commons

### 7.1 Is an ODR definition distinguishable from a plain weak definition?

`inl.h`
```c
inline int q(void) { return 42; }
```

`mo_inl.cpp`
```c
#include "inl.h"
int use_inl(void) { return q(); }
```

`mo_weak.cpp`
```c
__attribute__((weak)) int q(void) { return 7; }
int use_weak(void) { return 0; }
```

`mo_strong.cpp`
```c
int q(void) { return 99; }
int use_strong(void) { return 0; }
```

`mo_main2.cpp` calls `use_inl() + use_weak()`; `mo_main3.cpp` calls
`use_inl() + use_strong()`.

```
$ $NM -m mo_inl.o    | grep __Z1qv
0000000000000014 (__TEXT,__text) weak external __Z1qv
$ $NM -m mo_weak.o   | grep __Z1qv
0000000000000000 (__TEXT,__text) weak external __Z1qv
$ $NM -m mo_strong.o | grep __Z1qv
0000000000000000 (__TEXT,__text) external __Z1qv
```

```
$ run o_mq1 mo_main2.o mo_inl.o mo_weak.o          # ODR(=42) + weak(=7)
ld exit=0
$ $OD -d --disassemble-symbols=__Z1qv o_mq1
100000458: 52800540    mov w0, #0x2a    ; =42      # inline won (first)

$ run o_mq2 mo_main3.o mo_inl.o mo_strong.o        # ODR(=42) + strong(=99)
ld exit=0
$ $OD -d --disassemble-symbols=__Z1qv o_mq2
100000370: 52800c60    mov w0, #0x63    ; =99      # strong won
```

### Inference

**On Mach-O an ODR definition is behaviourally identical to a plain weak
definition.** The symbol table entries are byte-identical (`weak external` for
both), and resolution treats them the same: first-wins against another weak
definition (MQ1), loses to a strong definition (MQ2).

So Mach-O needs no `EntityCoalescing` rank at all — the field is inert for
resolution on this platform. It remains worth carrying in the summary because
it is front-end knowledge the object format discards, and because it gates the
summary-mismatch check (a differing summary between two ODR copies is an ODR
violation; between two plain weak copies it is legal).

Contrast COFF, where coalescing *is* the conflict predicate.

### 7.2 Visibility on common symbols — order-dependent

Section 2 established that two weak *definitions* merge to the least
restrictive visibility. Commons behave differently.

`hc_hidden_common.c`
```c
__attribute__((visibility("hidden"))) int g;
```

`hc_default_common.c`
```c
int g;
```

`hc_read.c`
```c
extern int g;
int MAINSYM(void) { return g; }
```

Compiled with `-fcommon -DMAINSYM=main`.

```
$ $RO --syms m_hc.o | grep -A3 "Name: _g"
    Name: _g (1)  PrivateExtern  Extern  Type: Undef (0x0)

$ ld -o m_hc_out  m_rd.o m_hc.o m_dc.o ...        # hidden common FIRST
ld exit=0
$ $NM -m m_hc_out | grep " _g$"
0000000100004000 (__DATA,__common) non-external (was a private external) _g

$ ld -o m_hc_out3 m_rd.o m_dc.o m_hc.o ...        # default common FIRST
ld exit=0
$ $NM -m m_hc_out3 | grep " _g$"
0000000100004000 (__DATA,__common) external _g

$ ld -o m_hc_out2 m_rd.o m_hc.o m_hc2.o ...       # both hidden
0000000100004000 (__DATA,__common) non-external (was a private external) _g

$ ld -o m_hc_out4 m_rd.o m_dc.o m_dc2.o ...       # both default
0000000100004000 (__DATA,__common) external _g
```

### Inference

**For common symbols, Mach-O visibility is order-dependent**: hidden-first
yields a private symbol, default-first yields an exported one. This is the
opposite of the weak-definition case in section 2, where `hidden + default`
gave an exported symbol in *either* order.

The rule is not a commutative merge at all — `addCommon`
(`lld/MachO/SymbolTable.cpp:251`) simply carries the `isPrivateExtern` of
whichever common it keeps, with no `&=` step. Only `addDefined`'s weak-def
coalescing path performs the `privateExtern &= isPrivateExtern` merge.

**Implication for our model.** We cannot faithfully reproduce this, because our
`mergeLinkage` is deliberately commutative and order-independent — a property
we want, since a link unit's linkage record should not depend on input order.
Emulating Mach-O exactly would make the LU summary order-dependent for this one
case.

The honest choice is to deviate deliberately and document it: apply the
least-restrictive rule uniformly on Mach-O (matching the weak-definition case,
which is by far the common one) and accept that a `hidden common + default
common` pair yields `Default` where ld-prime would give private if the hidden
one happened to be linked first. This is a case where being order-independent
is worth more to a summary format than bug-compatibility with a link-order
artifact.

> **Superseded.** The implementation plan (§2.6) now reproduces this faithfully
> and warns, rather than approximating commutatively. The LU is already
> order-dependent via `incomingDataWins`'s first-wins tie-break, so
> commutativity of `mergeLinkage` was never a guarantee about the link unit.

---

## 8. Fifth pass: two declarations with differing bindings

Same program as ELF §9 (`strongdecl.c`, `weakdecl.c`, `dmain.c`), compiled with
`-DMAINSYM=main`.

```
$ $NM m_sd.o | grep " _g$"
                 U _g
$ $NM m_wd.o | grep " _g$"
                 U _g               # NOT marked weak

$ ld -o m_d1 m_dm.o m_wd.o m_sd.o ...
ld exit=1
Undefined symbols for architecture arm64:
  "_g", referenced from:
      _use_weak_decl in m_wd.o
      _use_strong_decl in m_sd.o
```

### Inference

Both declarations produce a plain `U` — clang does not mark a weak *declaration*
weak on Mach-O at all, corroborating the same observation in §6.1. There is
consequently nothing to join: the merged binding is `Undefined` either way, and
the link fails because the symbol has no definition.

Mach-O agrees with ELF on the outcome (error) but for a different reason: on
ELF the strong reference dominates a genuinely-weak one, whereas here neither
reference was weak to begin with.



