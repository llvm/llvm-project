# COFF Linker Behavior — Grounding Probes

Probes run against `lld-link` (LLD 24.0.0, built from this tree) to ground the
`LinkageRules` for `Triple::COFF`. Target: `x86_64-pc-windows-msvc`.

Tools used:

- `CLANG=/Volumes/avirals-external-drive-1/aviralg-llvm-project/bin/clang`
- `CLANGXX=/Volumes/avirals-external-drive-1/aviralg-llvm-project/bin/clang++`
- `LINK=/Volumes/avirals-external-drive-1/aviralg-llvm-project/bin/lld-link`
- `NM=/Volumes/avirals-external-drive-1/aviralg-llvm-project/bin/llvm-nm`
- `RO=/Volumes/avirals-external-drive-1/aviralg-llvm-project/bin/llvm-readobj`

All objects compiled with:

```
$CLANG --target=x86_64-pc-windows-msvc -c -O0 <src> -o <obj>
```

All links use `/entry:mainCRTStartup /subsystem:console /nodefaultlib` to avoid
needing the MSVC runtime, which is not present on this host.

---

## 1. Binding precedence and the duplicate-definition error

### Program

`strong_a.c` → `int f(void) { return 1; }`
`strong_b.c` → `int f(void) { return 2; }`
`weak_a.c`   → `__attribute__((weak)) int f(void) { return 1; }`
`weak_b.c`   → `__attribute__((weak)) int f(void) { return 2; }`

`main.c`
```c
int f(void);
int mainCRTStartup(void) { return f(); }
```

### Commands and output

```
$ $NM strong_a.obj | grep " f$"
00000000 T f
$ $NM weak_a.obj | grep -i " f$"
00000000 W f
```

```
$ $LINK /out:out_c1.exe ... main.obj strong_a.obj strong_b.obj
lld-link: error: duplicate symbol: f
>>> defined at strong_a.obj
>>> defined at strong_b.obj

$ $LINK /out:out_c2.exe ... main.obj strong_a.obj weak_b.obj      # strong + weak
   (no output — success)

$ $LINK /out:out_c3.exe ... main.obj weak_a.obj strong_b.obj      # weak + strong
   (no output — success)

$ $LINK /out:out_c4.exe ... main.obj weak_a.obj weak_b.obj        # weak + weak
lld-link: error: duplicate symbol: .weak.f.default
>>> defined at weak_a.obj
>>> defined at weak_b.obj

$ $LINK /out:out_c1b.exe ... /force:multiple main.obj strong_a.obj strong_b.obj
lld-link: warning: duplicate symbol: f
>>> defined at strong_a.obj
>>> defined at strong_b.obj
```

### Inference

- **Strong + strong is an error**, as on ELF and Mach-O.
- **Strong + weak is fine**, as elsewhere.
- **Weak + weak is an ERROR on COFF.** This is a genuine three-way divergence:
  ELF probe P4 and Mach-O probe M4 both accepted two weak definitions and kept
  the first. COFF's `__attribute__((weak))` is emulated with an alias symbol
  (note the mangled name `.weak.f.default` in the diagnostic), and two such
  aliases collide.

  Consequence: the conflict predicate "both non-weak" is **wrong for COFF**.
  On COFF, weakness does not license duplicate definitions — only COMDAT does
  (see section 2).

- `/force:multiple` downgrades the error to a warning, the COFF analogue of
  ELF's `-z muldefs`, and matches our `WarnOnMultipleDefinitions` flag.

---

## 2. COMDAT: how ODR is represented, and what actually licenses duplicates

### Program

`inl.h`
```c
inline int inl(void) { return 42; }
```

`inl_a.cpp` / `inl_b.cpp` — each `#include "inl.h"` and call `inl()`.

`inlmain.cpp`
```c
int use_a(void); int use_b(void);
extern "C" int mainCRTStartup(void) { return use_a() + use_b(); }
```

For the COMDAT-vs-regular case:

`inl_comdat_only.cpp`
```c
inline int inl(void) { return 42; }
int use_c(void) { return inl(); }
```

`inl_regular.cpp`
```c
int inl(void) { return 99; }
```

`k4main.cpp`
```c
int use_c(void);
extern "C" int mainCRTStartup(void) { return use_c(); }
```

### Commands and output

The inline function's symbol class and section characteristics:

```
$ $NM inl_a.obj | grep -i "inl@@"
00000000 T ?inl@@YAHXZ                       # T = strong/external, NOT W

$ $RO --sections inl_a.obj | grep -A8 "IMAGE_SCN_LNK_COMDAT"
    Characteristics [ (0x60501020)
      IMAGE_SCN_ALIGN_16BYTES (0x500000)
      IMAGE_SCN_CNT_CODE (0x20)
      IMAGE_SCN_LNK_COMDAT (0x1000)
      IMAGE_SCN_MEM_EXECUTE (0x20000000)
      IMAGE_SCN_MEM_READ (0x40000000)
    ]

$ $RO --symbols inl_a.obj | awk '/Name: \.text/{f=1} f&&/Selection/{print; f=0}'
      Selection: 0x0
      Selection: Any (0x2)                   # IMAGE_COMDAT_SELECT_ANY
```

Two COMDAT copies coalesce silently:

```
$ $LINK /out:out_ci1.exe ... inlmain.obj inl_a.obj inl_b.obj
   (no output — success)
```

COMDAT against a *regular* definition of the same symbol:

```
$ $LINK /out:o_k4.exe ... k4main.obj inl_comdat_only.obj inl_regular.obj
lld-link: error: duplicate symbol: int __cdecl inl(void)
>>> defined at inl_comdat_only.obj
>>> defined at inl_regular.obj
```

### Inference

This is the most important COFF result for our model.

- **An ODR definition on COFF keeps `Strong` binding.** `nm` reports `T`, not
  `W`. Compare ELF (`W` + COMDAT group) and Mach-O (`WeakDef` flag). So the
  binding value emitted for `inline` is **target-dependent**: `Weak` on
  ELF/Mach-O, `Strong` on COFF.

- **COMDAT, not binding, is what licenses duplicate definitions on COFF.**
  Two COMDAT copies coalesce (CI1); COMDAT + regular is a duplicate error (K4);
  and from section 1, weak + weak is *also* an error. So the COFF conflict
  predicate is:

      conflict  <=>  NOT (both definitions are COMDAT)

  which is exactly `SymbolTable::addComdat` at `lld/COFF/SymbolTable.cpp:924`:
  `if (!existingSymbol->isCOMDAT) reportDuplicate(s, f);`

- **This is the strongest justification for splitting `EntityCoalescing` out of
  `EntityBinding`.** On COFF an ODR definition is `Strong` + `ODR`. A single
  ordered enum cannot represent it: as `Strong` alone, two inline definitions
  would wrongly conflict; as a `WeakODR` rank below `Strong`, the binding would
  wrongly lose to a regular definition instead of erroring. Only two
  independent fields give the right answer for both K4 and CI1.

- `Selection: Any (0x2)` is `IMAGE_COMDAT_SELECT_ANY`. COFF has five other
  selection kinds (`SAME_SIZE`, `EXACT_MATCH`, `ASSOCIATIVE`, `LARGEST`,
  `NODUPLICATES`), which is why modelling coalescing as an enum rather than a
  bool leaves room to grow.

---

## 3. Common symbols

### Program

`common_a.c` → `int g;`, `common_big.c` → `long long g[4];`,
`def_g.c` → `int g = 7;`, `weakdef_g.c` → `__attribute__((weak)) int g = 9;`,
`mainj.c` → `extern int g; int mainCRTStartup(void) { return g; }`

Compiled with `-fcommon`.

### Commands and output

```
$ for x in common_a common_big def_g weakdef_g; do $NM $x.obj | grep " g$"; done
00000004 C g          # common_a
00000020 C g          # common_big
00000000 D g          # def_g
00000000 W g          # weakdef_g
```

```
$ $LINK /out:o_k1.exe ... mainj.obj common_a.obj common_big.obj   # common + common
   (no output — success)
$ $LINK /out:o_k2.exe ... mainj.obj common_a.obj def_g.obj        # common + strong
   (no output — success)
$ $LINK /out:o_k3.exe ... mainj.obj weakdef_g.obj common_big.obj  # weak + common
   (no output — success)
```

### Inference

COFF accepts all three combinations without diagnostics, matching
`SymbolTable::addCommon` at `lld/COFF/SymbolTable.cpp:939`, which replaces the
existing symbol only when the incoming common is larger and otherwise silently
keeps what it has. Commons never conflict on any of the three platforms.

---

## 4. Undefined symbols

### Program

`undef_main.c`
```c
int missing(void);
int mainCRTStartup(void) { return missing(); }
```

### Commands and output

```
$ $LINK /out:o_ku1.exe ... undef_main.obj
lld-link: error: undefined symbol: missing
>>> referenced by undef_main.obj:(mainCRTStartup)

$ $LINK /out:o_ku2.exe ... /force:unresolved undef_main.obj
lld-link: warning: undefined symbol: missing
>>> referenced by undef_main.obj:(mainCRTStartup)

$ $LINK /dll /out:o_ku3.dll /noentry ... undef_main.obj
lld-link: error: undefined symbol: missing
>>> referenced by undef_main.obj:(mainCRTStartup)
```

### Inference

- An unresolved reference is an error, downgradable to a warning with
  `/force:unresolved` — a two-state policy, unlike ELF's three-state
  `--unresolved-symbols`.
- **Building a DLL does not relax the rule** (KU3), matching Mach-O and
  differing from ELF, where `-shared` accepted unresolved references. Windows
  DLLs must be fully resolved at link time.

So on undefined symbols, ELF is the outlier: two of three platforms require
full resolution regardless of the output artifact kind.

---

## 5. Visibility

COFF has no symbol visibility field at all. Export from a DLL is controlled by
`__declspec(dllexport)` / `.def` files, which is an unrelated mechanism
operating per-DLL rather than a per-symbol attribute merged during resolution.

There is therefore no COFF analogue of `EntityVisibility::Hidden` or
`::Protected`, and no visibility merge rule to emulate. A summary carrying
anything other than `Default` for a COFF target is not representable on the
platform.

---

## 6. Second pass: ordering details

Section 3 established that no combination of commons errors, but did not
establish *which* definition prevails. These probes settle the rank table.

### 6.1 Which definition wins: common vs strong vs weak

`k_read.c`
```c
extern int g;
int mainCRTStartup(void) { return g; }
```

Payloads are distinguishable: `def_g.c` has `g = 7`, `weakdef_g.c` has `g = 9`,
and commons are zero-initialised. `/map:` output names the winning object.

```
$ $LINK /out:o_k2.exe ... /map:k2.map k_read.obj common_a.obj def_g.obj
$ grep -iE "\bg\b" k2.map
 0002:00000000       g            0000000140002000     def_g.obj

$ $OD -s -j .data o_k2.exe
 140002000 07000000 00000000                    ........      # value 7 = strong def

$ $LINK /out:o_k3.exe ... /map:k3.map k_read.obj weakdef_g.obj common_big.obj
$ grep -iE "\bg\b" k3.map
 0002:00000000       .weak.g.default   0000000140002000     weakdef_g.obj
 0000:00000000       g                 0000000140002020     <common>

$ $LINK /out:o_k1.exe ... /map:k1.map k_read.obj common_a.obj common_big.obj
$ grep -iE "\bg\b" k1.map
 0000:00000000       g            0000000140002020     <common>
$ $OD -h o_k1.exe | grep .data
  1 .data         00000000 ...                                # commons live in .bss
```

Section sizes corroborate: `o_k2` has an 8-byte `.data` (the 4-byte strong
definition, padded), while `o_k3` has 0x40 — large enough for the 32-byte
common.

### Inference

- **Strong beats common** (K2: `g` resolves to `def_g.obj`, value 7).
- **Common beats weak** (K3: `g` resolves to `<common>`, and the weak
  definition survives only as the alias symbol `.weak.g.default`, unreferenced).
- **Two commons merge**, taking the larger (K1).

So COFF's binding order matches **ELF**, not Mach-O: `Common` outranks `Weak`.
Mach-O is the sole platform where a weak definition displaces a common.

### 6.2 Weak declaration + common definition

Same program as the ELF doc section 6.1, compiled with
`-fcommon -DMAINSYM=mainCRTStartup`.

```
$ $NM w_weakdecl.obj  | grep -i " g$"
00000000 W g
$ $NM w_commondef.obj | grep -i " g$"
00000004 C g

$ $LINK /out:w_out.exe ... /map:w.map w_rd.obj w_weakdecl.obj w_commondef.obj
$ grep -iE "\bg\b" w.map
 0002:00000000 00000008H .rdata$.refptr.g          DATA
 0000:00000000  .weak.g.default.use  0000000000000000  <absolute>
 0002:00000000  .refptr.g            0000000140002000  w_weakdecl.obj
 0000:00000000  g                    0000000140003000  <common>
```

**Inference.** `g` resolves to `<common>`; the weak declaration contributes only
an unreferenced alias. Same conclusion as ELF and Mach-O — the binding join is
scoped to defining occurrences.

### COFF rank tables

Derived from sections 1, 2, 3 and 6:

| `EntityBinding` | rank | evidence |
|---|---|---|
| `Undefined` | 0 | any definition beats an undefined reference (KU1) |
| `Weak` | 1 | common beats weak (K3) |
| `Common` | 2 | strong beats common (K2); common beats weak (K3) |
| `Strong` | 3 | strong wins everything (K2) |

Same as ELF.

| `EntityCoalescing` | rank | evidence |
|---|---|---|
| `None` | 0 | COMDAT + regular is an error, so neither "wins" — see below |
| `ODR` | 1 | COMDAT + COMDAT coalesces (CI1) |

`EntityVisibility`: not representable. Any value other than `Default` for a
COFF target has no meaning on the platform.

### Conflict predicate

COFF's is **not** derivable from the binding ranks. Collected results:

| case | result | probe |
|---|---|---|
| strong + strong | error | C1 |
| strong + weak | ok | C2/C3 |
| weak + weak | **error** | C4 |
| COMDAT + COMDAT | ok | CI1 |
| COMDAT + regular | error | K4 |
| common + anything | ok | K1/K2/K3 |

`conflict ⟺ NOT both ODR` fails on strong+weak; `conflict ⟺ neither is weak`
fails on weak+weak. The rule that fits every row is:

```
both are definitions
  AND neither is Common
  AND NOT (both are ODR)
  AND NOT (exactly one is Weak)
```

The weak clauses follow from COFF emulating weak symbols with an alias — the
diagnostic in C4 names `.weak.f.default`, so two weak definitions collide on
the alias while one weak plus one strong does not.

This is why `LinkageRules` must expose `isConflictingDefinition` as its own
virtual rather than deriving conflicts from the rank tables.

---

## 7. Third pass: visibility on common symbols

Section 5 asserted that COFF has no visibility concept. This probe confirms
that a `visibility("hidden")` attribute has no effect on a COFF target, rather
than being silently encoded somewhere.

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

Compiled with `-fcommon -DMAINSYM=mainCRTStartup`.

```
$ $NM w_hc.obj | grep -i " g$"
00000004 C g                     # identical class to the default-visibility one

$ $LINK /out:w_hc.exe ... /map:whc.map w_rd.obj w_hc.obj w_dc.obj
$ grep -iE "\bg\b" whc.map
 0000:00000000       g            0000000140002000     <common>
```

### Inference

The hidden common is indistinguishable from a default-visibility common in the
object file (both plain `C`), and clang emits no diagnostic. Visibility is
simply dropped on this target.

Confirms that `EntityVisibility` needs no rank table for COFF: any value is
equivalent to `Default`. A summary carrying `Hidden` or `Protected` for a COFF
target is a mis-extraction, since clang could not have produced it from source.

---

## 8. Fifth pass: weak declarations are not undefined references

Same program as ELF §9 (`strongdecl.c`, `weakdecl.c`, `dmain.c`), compiled with
`-DMAINSYM=mainCRTStartup`.

```
$ $NM w_sd.obj | grep -i " g$"
         U g                       # undefined
$ $NM w_wd.obj | grep -i " g$"
00000000 W g                       # weak external — NOT undefined

$ $RO --symbols w_wd.obj | grep -A10 "Name: g$"
    Name: g
    Section: IMAGE_SYM_UNDEFINED (0)
    StorageClass: WeakExternal (0x69)
    AuxSymbolCount: 1
    AuxWeakExternal {
      Linked: .weak.g.default.use_weak_decl (17)
      Search: Alias (0x3)
    }

$ $LINK /out:w_d1.exe ... /map:wd1.map w_dm.obj w_wd.obj w_sd.obj
   (no output — success, either order)
$ grep -iE "\bg\b" wd1.map
 0000:00000000       g                              0000000000000000  <absolute>
 0000:00000000       .weak.g.default.use_weak_decl  0000000000000000  <absolute>
```

### Inference

**COFF does not error here, unlike ELF and Mach-O.** A COFF weak external is
not an undefined reference: it carries an `AuxWeakExternal` record naming a
*default* alias to bind to when no real definition is found. The link succeeds
with `g` resolved to an absolute 0.

This is the same alias mechanism that makes two weak *definitions* collide
(§1, C4). COFF's `__attribute__((weak))` is emulation rather than a native
binding, and it behaves differently from ELF/Mach-O weakness in both
directions: more permissive for declarations, less permissive for definitions.

For our model this is a Phase 3 concern rather than a join concern — the merged
binding still follows `argmax strengthRank` — but it means undefined-symbol
reporting should not assume COFF would have errored where ELF does.



