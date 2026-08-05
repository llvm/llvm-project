# Design note: canonical-class comparison vs a generic attribute vocabulary

## Summary

We annotate a custom type by **declaring which standard class it behaves like**
(`[[clang::analyze_as_class("std::optional")]]`, methods mapped with
`analyze_as_method`). The alternative would be a **generic vocabulary** that
describes a type's semantics abstractly, so any check could consume it without
knowing about specific std classes.

We chose class-comparison because **the semantic knowledge each clang-tidy check
needs already exists inside the check, keyed to specific std classes.** Anchoring
a custom type to a canonical class lets it *inherit that existing knowledge by
identity*. A generic scheme would instead require lifting that knowledge out of
every check into an abstract vocabulary and re-expressing it — an expensive,
per-check, error-prone migration whose cost scales with the semantic richness of
each check.

## The key idea: where the knowledge lives

An attribute can only ever supply **identity** ("this type is X-like", "this
method plays role Y"). It cannot supply **behaviour** — the matchers, the
judgments, and the fix-its live in each check. So the real question is: how much
does a check already know, and how cheaply can a custom type tap into it?

- **Class-comparison** reuses the check's existing per-std-type logic wholesale.
  The custom type says "treat me as `std::set`" and the check's vetted `std::set`
  reasoning applies unchanged.
- **Generic vocabulary** discards that leverage: each check must be rewritten to
  reason from abstract properties, and those properties must first be designed to
  be rich enough to reconstruct what the std class already embodies.

## Worked examples of the cost gap

The two examples below sit at opposite ends of the spectrum, which is the point:
the generic cost is not uniform — it explodes with the check's semantic depth.

### Example 1 — `readability-container-size-empty` (the cheap end)

Rewrites `x.size() == 0` to `x.empty()`.

- Knowledge it needs: "which method is the boolean emptiness predicate."
- Generic version: define a role `empty`, teach the matcher to accept it
  (`ContainerSizeEmptyCheck.cpp:154` requires a bool method literally named
  `empty`), and make the fix-it emit the *matched method's real name* instead of
  the hardcoded `"empty()"` (`:275`-`281`).
- Class-comparison version: the custom type declares it is container-like and
  maps its predicate method; the same fix-it work is needed either way.

**Verdict:** here the gap is modest — the knowledge is a single name mapping, so
generic is merely *a bit more* work. If every check looked like this, generic
would be defensible.

### Example 2 — `performance-inefficient-algorithm` (the expensive end)

Flags `std::find(c.begin(), c.end(), x)` on an associative container (member
`c.find(x)` is faster) and rewrites it.

- What the check actually knows is a **hardcoded behavioural contract**, keyed on
  the std class name:
  - the associative container name list (`:31`-`34`: set/map/multiset/.../unordered_*),
  - which algorithms have faster members (`:29`),
  - and subtle per-type semantics derived by string-matching the name:
    `Unordered = name.contains("unordered")` (`:69`), `Maplike = ...("map")`
    (`:70`), and "unordered containers have no ordered-bound equivalent"
    (`:100`).
- Generic version: the attribute would have to **encode all of that as
  properties** — "this container has a member `find` asymptotically faster than
  linear scan," "it is ordered vs unordered," "it is map-like vs set-like," "it
  lacks `lower_bound`," etc. That is re-deriving `std::set`'s entire behavioural
  contract in attribute form, and putting its *correctness* on the annotator (a
  mis-declared complexity yields wrong advice).
- Class-comparison version: the custom type says `analyze_as_class("std::set")`
  and **inherits the whole contract for free** — the ordered/unordered, map/set,
  and bound-availability reasoning all apply unchanged, because the check already
  contains it.

**Verdict:** here the gap is enormous. The generic route means reconstructing
years of accumulated per-type special-casing as a general vocabulary; the
class-comparison route is a one-line declaration.

## Why the gap generalises

The `~dozen` candidate consumer checks (see `plan_general.md` §12) each embed
their own hardcoded per-std-type knowledge. Class-comparison reuses it N times for
the price of an identity declaration. A generic vocabulary must be designed to
capture the *union* of everything those checks care about (emptiness, lookup
complexity, iterator/ownership semantics, emplace-equivalence, ...) — an
open-ended surface — and then each check must be rewritten to consume it.

## Secondary advantages of class-comparison

- **Correctness by reuse.** It runs the checks' already-vetted std logic rather
  than trusting annotator-supplied semantic claims.
- **Bounded vocabulary.** Its "vocabulary" is essentially the names of standard
  classes — small and well-understood. A generic property vocabulary is
  open-ended and must grow with every consumer.
- **Incremental adoption.** A check opts in by honouring "custom → std::X" via one
  shared helper; no per-check semantic re-modelling.

## Honest limits (state these too)

- Class-comparison requires the custom type to genuinely mirror a std class, and
  its members to map to the std members (that is the `analyze_as_method` layer).
- A truly novel abstraction with no std analogue is not served — but that is rare
  and arguably out of scope.
- The fix-it must still emit the custom type's real method names; that cost is
  shared by both designs.

## Recommendation

Anchor to canonical classes. Keep the annotation as an **identity declaration**
that lets checks reuse their existing per-type knowledge, and treat cross-check
adoption as an incremental roadmap (`plan_general.md` §12), not a generic
rewrite. The `inefficient-algorithm` example is the clearest argument: the
knowledge that makes it work cannot be cheaply externalised into an attribute —
it *is* the std class.
