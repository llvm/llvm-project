# Plan: role-attribute approach (branch `generalAttributesKay`)

Kickoff plan for the *alternate* proposal. This branch is a clean-room from
`main` (no `analyze_as_*` string-match implementation). The goal is a **single,
closed role-attribute vocabulary** that models one per-object boolean predicate,
serving BOTH target types with the same engine.

Companion docs: `architecture.md` (how the optional pipeline works + the
levels-of-validation and two-worlds framing). The string-match MVP lives on
`attributesKay`; this is deliberately separate, not a rework.

Learning-exercise rule still applies: this plan is design/scope only. Do not
implement the compiler changes; fixtures (`hicketts/*_general.*`,
`hicketts_vector.*`) are fair game.

---

## 1. Thesis

Model each supported class as having **one named boolean predicate**, and let
method-level *role* attributes say how each method relates to it. This unifies:

- **optional** — predicate `engaged` (== today's `has_value`). Replaces the
  `analyze_as_method("...")` string keys with roles.
- **vector** — predicate `non_empty`. Fills the `precondition_gap` measured in
  the vector experiment (empty `front()`/`pop_back()` is UB and caught by
  *nothing* today — not the baseline, not Owner/Pointer).

Same dataflow question in both cases: "is the predicate established on this path
before a method that requires it?" That is exactly what the
`bugprone-unchecked-optional-access` model already answers for `has_value` — so
the core implementation idea is to **generalise that model from the hardcoded
`has_value` field to an arbitrary named predicate.**

## 2. Why roles beat verbatim signatures (recap of the decision)

- **Identity vs role.** The string signature described how to *identify* a method
  (its params). But overload resolution + the attribute sitting on one specific
  decl already identify it. What the model actually needs is the method's *role*.
- **Per-decl placement disambiguates overloads** — no signature strings needed.
- **Header-free.** Roles never resolve the real `std::optional`/`std::vector`, so
  `#include` is irrelevant (see `architecture.md` §4, the `<optional>` blocker).

## 3. Precedents to ride (all in-tree — cite these in the RFC)

- **Capability / thread-safety attributes** (`Attr.td:4101`–`4184`):
  `RequiresCapability`, `AcquireCapability`, `ReleaseCapability`. This is
  literally requires-valid / makes-valid / makes-invalid as a **closed,
  capability-scoped** role vocabulary — the exact shape we want, already accepted.
- **`reinitializes`** (`:4877`) — "returns object to a defined state"; already
  applies to both `optional::reset()` and `vector::clear()`.
- **Consumed / typestate attributes** (`:4285`–`:4366`) — `Consumable`,
  `SetTypestate`, `CallableWhen`, `TestTypestate`. Cite as evidence typestate is
  acceptable in clang, but frame OUR proposal as the *narrower capability shape*,
  NOT general typestate, to avoid the earlier rejection.

## 4. Scope boundary (state it up front in the RFC)

Role attributes model a **single per-object state predicate**. In scope:

- optional `engaged`; vector `non_empty`; reset-to-valid (`reinitializes`).

Explicitly OUT of scope (and why):

- **Relational / aliasing hazards** (iterator invalidation: `push_back` stales
  existing iterators). Not a per-object bit — stays with `Owner`/`Pointer`
  (which already handle it; see the vector experiment: dangling *was* caught).
- **Numeric invariants** (`size`/`capacity` relationships). The model tracks a
  predicate, not a quantity.

## 5. Measured motivation (vector experiment, recorded here)

`test_hicketts_vector.cpp`, built with `build-llvm/bin/clang++`
`--target=arm64-apple-darwin -std=c++17`:

| Case | Baseline (attrs off) | Owner/Pointer + lifetimebound |
|------|----------------------|-------------------------------|
| `front()` of a temporary (dangling ref) | silent | ⚠ `-Wdangling` |
| `begin()` of a temporary (dangling iter) | silent | ⚠ `-Wdangling` |
| ref/iter into a live vector | silent | silent ✓ |
| **`front()` on an EMPTY vector (UB)** | **silent** | **silent** ← the gap |

The empty-access row is what the `requires_state("non_empty")` role must make
warn.

## 6. Proposed vocabulary (DRAFT — open for iteration)

Two axes still open; capture both, pick during RFC:

- **Predicate naming:** string (`"engaged"`, `"non_empty"`) vs a fixed enum vs a
  single implicit predicate per class. Capability analysis names its capability,
  so a small **closed string/enum** is precedented and probably best.
- **Class-level opt-in:** reuse an `analyze_as_class`-style marker to declare the
  class is state-tracked and name its predicate.

Draft method roles (map straight onto existing optional transfer functions):

| Role (draft spelling) | Meaning | optional example | vector example | model action |
|---|---|---|---|---|
| `requires_state("P")` | precondition: P must hold, else warn | `value()`/`unwrap()` | `front()`/`pop_back()` | diagnose if P not established |
| `sets_state("P")` | establishes P true | value ctor, `emplace` | `push_back` | set predicate true |
| `clears_state("P")` | establishes P false | nullopt ctor, `reset` | `clear` (+`reinitializes`) | set predicate false |
| `queries_state("P")` | narrows P in flow | `has_value`/`operator bool` | `empty()` | branch-sensitive refine |

Note: `requires`/`sets`/`clears`/`queries` ≈ `REQUIRES`/`ACQUIRE`/`RELEASE`/(test)
from thread-safety — keep the analogy explicit.

## 7. Validation strategy (from architecture.md §4)

- **L1** validate the role/predicate name against a closed table (typo-catch),
  in Sema at parse time — header-free.
- **L2** validate the annotated method's arity/shape if useful — also parse-time
  (the method decl is available to the Sema handler).
- **L3** (verify custom param types vs the real std type) — skip; needs the
  header and buys little. This is the whole point of NOT going the Plan A route.

## 8. Implementation sketch (design only — do NOT build yet)

1. `Attr.td` — add the class-level predicate marker + the method role attributes
   (model on the capability attribute defs at `:4101`+).
2. Sema — handlers + L1/L2 validation (mirror `handleAnalyzeAs*`; capability
   handlers are a closer template).
3. Model — the crux: generalise `UncheckedOptionalAccessModel` so the synthetic
   boolean field is a *named predicate* rather than hardcoded `has_value`
   (`:1330`, `:441`–`:445`), and drive the match-switch cases from the role
   attributes instead of hardcoded method names.
4. Decide: extend `bugprone-unchecked-optional-access` to arbitrary predicates,
   or spin a sibling check for the general "state precondition" analysis. (Open.)

## 9. Test plan

- **optional** — re-annotate `hicketts_optional_general.h` with the new roles;
  `test_hicketts_optional_general.cpp` should reproduce the MVP's behaviour
  (the same set of expected warnings/silences the string-match version produced).
- **vector** — add empty-access cases to `test_hicketts_vector.cpp`; the
  `precondition_gap` case must now warn, while the dangling cases keep warning
  via Owner/Pointer and safe cases stay silent.

## 10. Open questions

- Predicate naming: string vs enum vs fixed-per-class.
- One predicate per class, or several (e.g. a type with two independent states)?
- Diagnoser wording for "required state not established here."
- Extend the optional check vs new check (§8.4).
- Does branch-sensitive `queries_state` need more than the optional model already
  does for `has_value`/`operator bool`?

## 11. Files (this branch)

- `hicketts/hicketts_optional_general.h` / `test_hicketts_optional_general.cpp`
  — optional fixture (currently still carries old `analyze_as_*`; to be re-annotated).
- `hicketts/hicketts_vector.h` / `test_hicketts_vector.cpp` — vector fixture
  (Owner/Pointer + lifetimebound live; proposed roles commented).
- `hicketts/architecture.md` — pipeline map + design framings.
- (later) `Attr.td`, Sema, model changes in the real tree.
