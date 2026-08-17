# SSAF Entity Linker — Implementation Plan

Collected from the design discussion and grounded in the probe results in
`ssaf-linker-elf-behavior.md`, `ssaf-linker-macho-behavior.md` and
`ssaf-linker-coff-behavior.md`.

Nothing in phases 1–5 has been started. Phase 0 is already in the working tree.

---

## Phase 0 — already landed (working tree, uncommitted)

For context, since later phases modify this code:

- `WarnOnMultipleDefinitions` constructor flag (default: fail) plus
  `clang-ssaf-linker --warn-on-multiple-definitions`. On conflict the incoming
  occurrence is ignored entirely and the first definition stands.
- Reconciliation split into `isConflictingDefinition`, `incomingDataWins`,
  `mergeLinkage`, with `reportIfLinkageIsNotExternal` /
  `reportIfDefinitionsConflict` guards.
- Stateless pipeline: `resolve()` returns
  `pair<EntityResolutionMap, DataSelectionMap>`; no per-link member state.
- `EntityLinker` public surface reduced to constructor, `link()`,
  `takeOutput()`; reconciliation rules private with `friend class TestFixture`.
- Readability pass: `fatal()`, `lookupOrFatal()`, `insertOrFatal()`,
  `isExternal()`, `mergeEntityData()`, `mergeSummaryData()`,
  `checkTUNotAlreadyLinked()`, invariant asserts, section banners.
- Lit coverage for both multiple-definition modes; `tu-2.json` corrected so the
  shared external entity is a declaration rather than a second definition.

**Loose end:** `clang/test/Analysis/Scalable/ssaf-linker/Inputs/tu-redefines-shared-ext.json`
is untracked. It must be `git add`-ed or the two multiple-definition lit cases
fail for everyone else.

**To be reverted in Phase 1:** the explicit `= 0,1,2…` numbering added to
`EntityBinding`, `EntityVisibility` and `EntityDefinitionKind`, and the
`std::max` joins that depend on it. Superseded by the decision to move all
ordering into `LinkageRules`.

---

## Phase 1 — split the axes, remove orderings from the enums

### 1.1 `Model/EntityLinkage.h`

Drop `WeakODR`; add `EntityCoalescing`; strip all numeric values and
"ordered by…" wording. Every enum keeps `: uint8_t` for size only.

```cpp
/// Symbol scope. Enumerator order is insignificant: the linker never compares
/// these values, and namespace resolution switches on them by name.
enum class EntityLinkageType : uint8_t { None, Internal, External };

/// Symbol strength: which definition prevails when two collide.
///
/// Enumerator order is insignificant. Precedence differs per platform — on
/// Mach-O a weak definition displaces a common, on ELF and COFF the common
/// wins — so comparisons must go through LinkageRules::strengthRank().
enum class EntityBinding : uint8_t { Undefined, Weak, Common, Strong };

/// How far a symbol escapes its link unit.
///
/// Enumerator order is insignificant. ELF merges to the most restrictive,
/// Mach-O to the least, and COFF has no visibility at all; see
/// LinkageRules::visibilityRank().
enum class EntityVisibility : uint8_t { Default, Hidden, Protected };

/// Whether every definition of the entity is required to be identical, as a
/// COMDAT group (ELF/COFF) or .weak_definition (Mach-O) guarantees.
///
/// Independent of EntityBinding: an inline function is Weak+ODR on ELF and
/// Mach-O but Strong+ODR on COFF. Modelled as an enum rather than a bool so
/// COFF's other IMAGE_COMDAT_SELECT_* kinds can be added later.
enum class EntityCoalescing : uint8_t { None, ODR };

/// Whether this occurrence defines the entity or merely declares it.
enum class EntityDefinitionKind : uint8_t { Declaration, Definition };
```

`EntityLinkage` gains a fifth field `Coalescing`; constructor arity 4 → 5;
update `operator==`, `operator<<`, and add
`FIELD(EntityLinkage, Coalescing)` to `Model/PrivateFieldNames.def`.

### 1.2 `Core/ModelStringConversions.h`

Remove the `WeakODR` to/from-string cases. Add
`entityCoalescingToString` / `entityCoalescingFromString` ("None" / "ODR").

### 1.3 Serialization — `JSONFormat/JSONFormatImpl.{h,cpp}`

Add a **required** `"coalescing"` key alongside `"binding"`, `"visibility"`,
`"definition"`, with an `InvalidEntityCoalescing` error message following the
existing pattern.

> **Breaking format change.** Any summary produced before this lands becomes
> unreadable. There is no version field in the JSON schema to negotiate this.

### 1.4 Extractor — `TUSummary/TUSummaryExtractor.cpp`

**Target-independent.** The extractor records the *source-level* fact; the
platform lowering happens in `LinkageRules` (§2.2).

The three formats encode identical C++ semantics — `inline` means one entity,
defined identically in every TU, duplicates permitted — and differ only in how
they express it. COFF's encoding is the semantically faithful one: the binding
stays `Strong` and a separate COMDAT mechanism licenses the duplicates. ELF and
Mach-O have no general coalescing mechanism, so they *overload weak binding* to
obtain duplicate tolerance. That is an encoding trick, not a claim that inline
functions are interposable, so it does not belong in the summary.

Normalized mapping, identical for every target:

| source construct | `Binding` | `Coalescing` |
|---|---|---|
| ordinary definition | `Strong` | `None` |
| `inline` / template / vtable (`GVA_*ODR`) | `Strong` | `ODR` |
| `__attribute__((weak))` | `Weak` | `None` |
| C tentative definition | `Common` | `None` |
| declaration only | `Undefined` | `None` |

So `mapBinding` changes only by returning `Strong` where it previously returned
`WeakODR`, and a new `mapCoalescing(const NamedDecl *, ASTContext &)` returns
`ODR` for the two `GVA_*ODR` linkages. `__attribute__((weak))` continues to be
tested first, before the ODR switch.

Visibility needs no target awareness either: clang's AST is already
target-configured. The Mach-O probe showed `visibility("protected")` producing
`-Wunsupported-visibility` and being downgraded at compile time, so
`ND->getVisibility()` already returns a target-legal value.

### 1.5 Fixtures

67 JSON files under `clang/test/Analysis/Scalable/` contain `"binding"` and
need a `"coalescing"` key added. Purely additive, scriptable; per the
fixture-editing convention, insert in place without reordering existing keys.

### 1.6 Tests

`EntityLinkageTest.cpp` and `EntityLinkerTest.cpp`: 5-argument construction,
updated stream/format assertions, updated `def`/`decl` helpers.

---

## Phase 2 — `LinkageRules`

### 2.1 Interface — `Core/EntityLinker/LinkageRules.{h,cpp}`

```cpp
/// The platform-specific symbol resolution rules the linker emulates.
///
/// Ranks answer "which value wins the join", not "which value is semantically
/// greater". They are deliberately inverted for some fields on some targets.
class LinkageRules {
public:
  virtual ~LinkageRules() = default;

  /// Returns the rules for \p T, dispatching on its object format.
  static const LinkageRules &forTarget(const llvm::Triple &T);

  virtual llvm::StringRef getName() const = 0;

  /// Lowers a source-level (binding, coalescing) pair to the binding this
  /// platform actually emits. ELF and Mach-O have no general coalescing
  /// mechanism and encode ODR definitions as weak; COFF keeps them strong and
  /// licenses duplicates via COMDAT. All ranking and conflict detection
  /// operates on the result of this call, never on the raw binding.
  virtual EntityBinding effectiveBinding(EntityBinding B,
                                         EntityCoalescing C) const = 0;

  virtual unsigned strengthRank(EntityBinding) const = 0;
  virtual unsigned visibilityRank(EntityVisibility) const = 0;
  virtual unsigned coalescingRank(EntityCoalescing) const = 0;
  virtual unsigned definitionKindRank(EntityDefinitionKind) const = 0;

  /// Not derivable from the ranks — see the COFF table below.
  virtual bool isConflictingDefinition(const EntityLinkage &Current,
                                       const EntityLinkage &Incoming) const = 0;

  /// Normalizes \p Linkage to what this platform can represent.
  ///
  /// Values the platform silently drops are coerced to their platform
  /// equivalent; values the platform's toolchain could never have produced are
  /// a fatal error, since they can only come from a corrupted or hand-edited
  /// summary. See §2.7.
  virtual EntityLinkage normalize(const EntityLinkage &Linkage,
                                  const EntityName &Name) const = 0;
};
```

`forTarget` dispatches on `Triple::getObjectFormat()`; unsupported formats are
a fatal error naming the triple.

### 2.2 `effectiveBinding` — the platform lowering

| target | `(Strong, ODR)` | everything else |
|---|---|---|
| ELF | `Weak` | identity |
| Mach-O | `Weak` | identity |
| COFF | `Strong` | identity |

Evidence: ELF §3 (`inline` → `W` + COMDAT group), Mach-O §4 (`WeakDef` flag),
COFF §2 (`T` + `IMAGE_SCN_LNK_COMDAT`, `Selection: Any`).

This single virtual is what keeps the extractor target-independent, and it
reproduces every observed ODR-collision result (ELF §8):

| case | effective bindings | predicate | observed |
|---|---|---|---|
| ELF: ODR + regular | `Weak` vs `Strong` | no conflict | EI1 links, regular wins |
| ELF: ODR + ODR | `Weak` vs `Weak` | no conflict | EI2 links |
| Mach-O: ODR + regular | `Weak` vs `Strong` | no conflict | MI1 links, regular wins |
| COFF: ODR + regular | `Strong` vs `Strong`, not both ODR | conflict | K4 errors |
| COFF: ODR + ODR | `Strong` vs `Strong`, both ODR | no conflict | CI1 links |


### 2.3 Rank tables (all evidence-backed)

**Binding** — Mach-O inverts `Weak`/`Common`:

| | ELF | Mach-O | COFF |
|---|---|---|---|
| `Undefined` | 0 | 0 | 0 |
| `Weak` | 1 | **2** | 1 |
| `Common` | 2 | **1** | 2 |
| `Strong` | 3 | 3 | 3 |

Evidence: ELF §2 C3 (common beats weak), Mach-O §3 MC3 (weak beats common,
with a warning), COFF §6.1 K3 (common beats weak).

**Visibility** — Mach-O inverts the direction entirely:

| | ELF | Mach-O | COFF |
|---|---|---|---|
| `Default` | 0 | **1** | 0 |
| `Protected` | 1 | n/a | 0 |
| `Hidden` | 2 | **0** | 0 |

Evidence: ELF §4 V1 + §6.2 VP1/VP2 (most restrictive wins,
`Default < Protected < Hidden`), Mach-O §2 W1–W3 (least restrictive wins),
COFF §7 (visibility dropped entirely, so all ranks equal).

**Coalescing** — ranked so that a plain `max` yields the conservative meet:
`ODR` survives only if *every* copy is ODR.

| | all targets |
|---|---|
| `ODR` | 0 |
| `None` | 1 |

This looks inverted and is intentional: ranks are join-winners, and "not
guaranteed identical" must win over "guaranteed identical".

**DefinitionKind** — uniform: `Declaration` 0, `Definition` 1.

### 2.4 Conflict predicates

All predicates operate on `effectiveBinding(Binding, Coalescing)`, never on the
raw binding.

ELF and Mach-O:

```
both are definitions AND neither effective binding is Weak
```

An ODR definition lowers to `Weak` on these targets, so it never conflicts —
matching ELF §8 EI1/EI2 and Mach-O §8 MI1 — without the predicate mentioning
coalescing at all.

COFF — not derivable from ranks. From the six observed rows in COFF §6:

```
both are definitions
  AND neither is Common
  AND NOT (both are ODR)
  AND NOT (exactly one is Weak)
```

The weak clauses follow from COFF emulating weak symbols with an alias
(`.weak.f.default` appears in the C4 diagnostic), so two weak definitions
collide on the alias while one weak plus one strong does not. Note this is
about COFF's lowering of `__attribute__((weak))`, and is unrelated to how ODR
definitions are represented.

### 2.5 Rewiring `EntityLinker`

- Constructor computes `const LinkageRules &Rules = LinkageRules::forTarget(TargetTriple)`
  and stores the reference.
- The three reconciliation functions stay `static` but take
  `const LinkageRules &` as their first parameter. This keeps them pure and
  lets the unit tests sweep all three platforms. `TestFixture`'s accessors gain
  the same parameter.
- `mergeLinkage` becomes rank-driven, with the **definition-kind gate** the
  probes established (ELF §6.1, Mach-O §7.1, COFF §6.2 — all three agree a
  declaration contributes nothing to the binding):

```
Visibility     = argmax visibilityRank over both occurrences
DefinitionKind = argmax definitionKindRank over both occurrences

if both are definitions:
    Binding    = argmax strengthRank over both
    Coalescing = argmax coalescingRank over both
elif exactly one is a definition:
    Binding, Coalescing = that occurrence's
else:  // two declarations
    Binding    = argmax strengthRank over both
    Coalescing = None
```

Without the gate, joining a weak *declaration* with a common *definition* on
Mach-O would yield `Weak`, which no linker does.

- `incomingDataWins` uses `strengthRank` instead of `>` on the enum.
- `isConflictingDefinition` delegates to `Rules`.
- The `mergeLinkage` same-linkage-type assert stays.

Each `LinkageRules` subclass carries a comment with the minimal reproducer from
the probe docs justifying its rule, per the agreed convention.

### 2.6 Mach-O: order-dependent visibility on commons

**Mach-O visibility on commons is order-dependent** (Mach-O §7.2):

| inputs | ld-prime result |
|---|---|
| hidden common first, then default | private |
| default common first, then hidden | exported |
| both hidden | private |
| both default | exported |

The merged visibility is simply that of the **first** common. `addCommon`
(`lld/MachO/SymbolTable.cpp:251`) carries the `isPrivateExtern` of whichever
common it keeps; only `addDefined`'s weak-def coalescing path performs the
`privateExtern &= isPrivateExtern` merge.

**Decision: reproduce it faithfully, and warn.**

An earlier draft proposed approximating this commutatively to keep
`mergeLinkage` order-independent. That is the wrong trade. The LU is *already*
order-dependent — `incomingDataWins` keeps the first definition on ties, which
every platform does (ELF P4, Mach-O M4, COFF C4) — so commutativity of
`mergeLinkage` was a local property, not a guarantee about the link unit.
Approximating would produce `Default` where ld-prime gives private whenever the
hidden common is linked first, which is simply a wrong answer for any analysis
consuming the summary.

Because the case indicates genuine ambiguity in the user's program, it is also
reported:

```
warning: visibility of common symbol 'g' differs between translation units;
         the linked result depends on link order
```

Warned by default, unlike the COFF coercion in §2.7 — two commons disagreeing
on visibility is rare and always worth surfacing, whereas COFF visibility
coercion would fire on every portable header.

#### Interface consequence

A plain `visibilityRank` cannot express this, because the rule depends on the
*bindings*, not just the visibilities. So visibility gets a merge function with
a rank-based default:

```cpp
  /// Merges the visibility of two occurrences. The base implementation takes
  /// the higher visibilityRank(); MachOLinkageRules overrides it for commons.
  virtual EntityVisibility mergeVisibility(const EntityLinkage &Current,
                                           const EntityLinkage &Incoming) const;

  /// True if this pair's merge depends on link order, so the caller can warn.
  virtual bool isOrderDependentMerge(const EntityLinkage &Current,
                                     const EntityLinkage &Incoming) const;
```

- ELF, COFF: base implementation, `isOrderDependentMerge` always false.
- Mach-O: if both occurrences are `Common` definitions, return
  `Current.Visibility`; otherwise the base implementation.
  `isOrderDependentMerge` returns true when both are `Common` definitions and
  their visibilities differ.

`resolveEntity` calls `reportIfOrderDependentMerge(...)` alongside the existing
`reportIfDefinitionsConflict(...)` guard.

#### Test consequence

`mergeLinkage` loses its blanket "commutative, associative and idempotent"
doc-comment claim. The unit-test sweeps become per-platform, and for Mach-O the
commutativity sweep must exclude — or better, explicitly assert the
order-dependence of — pairs of commons with differing visibility. Idempotence
still holds everywhere (merging a linkage with itself never hits the divergent
case, since equal visibilities give the same answer in both orders).


### 2.7 Inexpressible values — `normalize()`

Not every `EntityLinkage` a summary can carry is representable on every target.
The behaviour differs by case, because *why* a value is unrepresentable differs.

Probed by dumping the AST for the same source across all three targets
(`clang -Xclang -ast-dump`, ELF doc §8 workspace):

| target | `Hidden` in AST | `Protected` in AST | diagnostic |
|---|---|---|---|
| ELF | `Hidden` | `Protected` | — |
| Mach-O | `Hidden` | **`Default`** | `-Wunsupported-visibility` |
| COFF | **`Hidden`** | **`Protected`** | **none** |

Two distinct situations follow:

**COFF `Hidden` / `Protected` → coerce to `Default`.** Clang accepts both
silently on Windows and drops them at emission (COFF §7: a hidden common is an
indistinguishable plain `C`). Portable code carries these attributes
unconditionally:

```c
__attribute__((visibility("hidden"))) int helper(void) { return 1; }
```

so a faithful COFF extraction legitimately contains `Hidden`. Rejecting it
would fail valid input, and would do so in the linker — far from the cause,
about a value the compiler never complained about. Coercion mirrors exactly
what the object file does. Report it only under the existing verbose flag; a
default-on warning would fire on every portable header.

**Mach-O `Protected` → fatal error.** Clang cannot produce it — it warns and
downgrades to `Default` at compile time — so its presence in a summary means
the summary was hand-edited or corrupted. That is precisely the case where a
loud failure is wanted. Message should name the entity and the target, and use
the non-prefixed style of `MultipleDefinition` rather than the
`EntityLinkerFatalErrorPrefix` style, since it is a bad-input error rather than
a logic bug.

`normalize()` returns the adjusted linkage and is called once per occurrence in
`resolveEntity`, before any ranking or conflict detection, so that all
downstream rules see only platform-legal values.

Note this keeps `mapVisibility` in the extractor target-independent (§1.4): the
extractor records what the source said, and the linker normalizes for the
target. That preserves the property that one extraction can be reinterpreted
for a different target.

### 2.8 Target triple validation

**Required by this phase, not optional.** `LinkageRules::forTarget` selects one
rule set from `Output.TargetTriple` and applies it to every linked TU. If a TU
was extracted for a different target, the linker silently applies the wrong
rules — and since COFF and Mach-O disagree on both the conflict predicate and
the `Weak`/`Common` order, the result would be wrong with no diagnostic.

`TUSummaryEncoding` already stores the triple and exposes `getTargetTriple()`,
and `EntityLinker`'s constructor already documents "every linked TU must report
the same triple". This closes the gap between that claim and the behaviour.

Add to `checkTUNotAlreadyLinked`, or a sibling guard called from `link()`
before `resolve()`:

```cpp
llvm::Error EntityLinker::checkTUTargetMatches(const llvm::Triple &TUTriple);
```

**Comparison strictness.** Exact string equality is too strict — it would
reject `arm64-apple-macosx14.0` against `arm64-apple-macosx15.0`, which are
identical for linkage purposes. Compare only the components that select rules
and affect resolution: architecture, vendor, OS and object format, ignoring the
version. This mirrors the existing key used by `MultiArchSharedLibrary` and
`MultiArchStaticLibrary`, which sort members on
`(getArch(), getSubArch(), getVendor(), getOS(), getEnvironment(), getObjectFormat())`.

Returns `llvm::Error`, not a fatal error: it is bad input, and `link()` already
returns errors for the analogous duplicate-TU-namespace case. Message should
name both triples and the offending TU namespace.

**Interaction with `clang-ssaf-linker`.** The tool currently hardcodes
`llvm::Triple("arm64-apple-macosx")` when constructing the linker, with a TODO
noting that architecture tracking is deferred. With this check in place, that
hardcoded triple would reject every non-Mach-O summary. The tool must instead
derive the LU triple from the first input summary and let the check validate
the rest — a small change to `runLink`, but a required part of this phase.

---

## Phase 3 — finalization: undefined symbols and hidden demotion

Both rules need the *complete* LU, since a later TU can change the answer — a
hidden declaration in TU5 demotes a definition from TU1 (ELF §4 V1). So they
run once at end of link, not per TU. Add a private `finalize()` called from
`takeOutput()`.

### 3.1 Undefined-symbol reporting

For each LU entity whose merged `DefinitionKind` is `Declaration`, report under
a tri-state policy (`Ignore` / `Warn` / `Error`), mirroring ELF's
`--unresolved-symbols`. Suppress when the merged binding is `Weak` — an
undefined weak reference is legal by design (ELF §5 U2).

**Default: `Ignore`.** The per-target defaults would be actively wrong here. A
real linker keys strictness on the artifact kind — ELF allows unresolved
references in a shared library but not an executable (§5 U1 vs U3), while
Mach-O and COFF reject them even for dylib/DLL (Mach-O §5 MU3, COFF §4 KU3). An
SSAF link unit is almost always an *intermediate* artifact, so defaulting to
strict would fire constantly on correct input. Expose
`--unresolved-symbols=ignore|warn|error` and let the caller assert completeness.

> **Limitation to document:** we do not track references. "Declaration
> everywhere" is inferred from the entity's presence in the id table, which
> means some TU mentioned it — close to, but not identical to, "referenced".

### 3.2 Hidden blocks cross-LU resolution

The faithful analogue of ELF turning a hidden symbol `LOCAL` and dropping it
from `.dynsym`: for each LU entity whose merged linkage is `External` and
merged visibility ranks as hidden on the target, rewrite its `LinkageTable`
entry to `Internal`.

Demote the **linkage type only, never the name**. Renaming would invalidate
every `EntityId` already patched into the summary data. It is also unnecessary:
external names are already LU-qualified, so `(f, LU1)` and `(f, LU2)` cannot
collide.

> **No consumer yet.** Nothing currently links LU summaries to each other —
> `MultiArch*` only bundles. So this is unobservable until an LU-combining
> stage exists. It is still worth doing now: it is the step that makes
> `Visibility` load-bearing rather than write-only, and doing it at
> finalization is the only correct point.

---

## Phase 4 — dropped-data mismatch check (on your signal)

Previously parked over the multi-format `equals` question; `EntityCoalescing`
now resolves *when* to check, and the format question still needs an answer.

When a duplicate external entity's summary data is dropped, patch the dropped
encoding into LU id space, compare it against the retained one, and report
mismatches. Gate on **`Coalescing == ODR`**: for ODR entities all copies are
required to be identical, so a difference is a real ODR violation or an
extraction bug. For plain `Weak`, differing definitions are legal and the same
check would be pure noise.

Mechanics settled earlier: stash dropped encodings during `merge`, add the ones
from the incoming TU to `PatchTargets` so they are patched uniformly, and
compare after `patch()` completes. Never patch twice.

Still open: `EntitySummaryEncoding::equals` needs to establish the concrete type
without RTTI (`LLVM_ENABLE_RTTI=OFF`). Options previously laid out: a per-class
type tag virtual (recommended), an unchecked `static_cast` with a documented
precondition, or LLVM-style RTTI with a closed kind enum.

---

## Phase 5 — lit fixtures

Three small hand-written fixtures, split by rule rather than by permutation,
since the reconciliation unit tests already cover the algebra exhaustively:

1. **`linkage-carry-through.test`** — one TU, ~6 entities, every enum value
   appearing at least once in a realistic combination, verifying each survives
   read → link → write.
2. **`join-winner.test`** — two TUs, ~6 entities, one per branch of the
   definition-kind × binding decision, each carrying a distinguishing summary
   blob so the output proves which TU's data survived.
3. **`join-visibility.test`** — two TUs, the visibility meets, with binding
   pinned so the fixture stays legal.

With `LinkageRules` in place these should be parameterized by target triple, so
the same entity shapes are checked against all three rule sets — in particular
the Mach-O binding inversion and the COFF COMDAT conflict rule.

---

## Suggested order

1. Phase 1 (model) — everything else depends on the enum shape.
2. Phase 2 (`LinkageRules`) — no point writing rule tables against an enum
   that is about to change.
3. Phase 5 fixtures for phases 1–2.
4. Phase 3 (finalization) — independent, additive.
5. Phase 4 (mismatch check) — on your signal.

Phases 1 and 2 are one atomic change from the tests' point of view: between
them the linker has no ordering to work with.

---

## Open questions blocking a clean start

All resolved.

- **Inexpressible values** — coerce on COFF, fatal for Mach-O `Protected`
  (§2.7).
- **Mach-O common visibility** — reproduce the order-dependence faithfully and
  warn (§2.6).
- **Target triple validation** — required, since it selects the rule set
  (§2.8).
- **Two declarations with differing bindings** — probed (ELF §9, Mach-O §8,
  COFF §8). `argmax strengthRank` confirmed; ELF shows the merged undefined
  symbol is `GLOBAL`, so a strong reference dominates a weak one in either
  order.

One finding from the last probe to carry into Phase 3: **a COFF weak
declaration is not an undefined reference.** It is a `WeakExternal` with an
`AuxWeakExternal` default-alias record, so COFF links successfully where ELF
and Mach-O both error (COFF §8). Undefined-symbol reporting must not assume
every platform would have failed where ELF does.

