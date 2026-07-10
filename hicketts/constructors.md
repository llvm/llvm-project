# Handling Overloads and Constructors in `analyze_as_method`

## Problem

The current `analyze_as_method("name")` attribute identifies target methods by name alone. This is insufficient when the target class has overloaded methods with different semantics — e.g. `set()` (resets) vs `set(T val)` (assigns). Since `analyze_as_class` is being generalised beyond `std::optional` to support any standard class, overloads are common and unavoidable.

Constructors are a special case of the same problem: they share a name but have fundamentally different semantics depending on their parameters (e.g. `optional(nullopt_t)` vs `optional(T&&)` vs `optional(in_place_t, Args...)`).

## Design Constraints

- The annotated (client) method must not be constrained to have the same parameter types as the target method. The whole point of annotations is that the custom class can have a completely different interface.
- The solution must work for both regular method overloads and constructors.
- The attribute should be the single source of truth for which target overload is intended.

## Proposed Solution: Signature Strings

Extend the `analyze_as_method` attribute string to include a parenthesised parameter list when disambiguation is needed:

```cpp
class [[clang::analyze_as_class("std::vector")]] MyVec {

    // Unambiguous — no signature needed
    [[clang::analyze_as_method("clear")]]
    void wipe();

    // Ambiguous — signature required
    [[clang::analyze_as_method("insert(iterator, const T&)")]]
    void shove_it_in(MyWeirdIter pos, MyT val, bool log = false);

    [[clang::analyze_as_method("insert(iterator, size_type, const T&)")]]
    void fill_insert(MyWeirdIter pos, int count, MyT val);
};
```

For constructors, the same mechanism applies — constructors are just methods named with the class name or a reserved keyword:

```cpp
class [[clang::analyze_as_class("std::optional")]] MyOpt {

    [[clang::analyze_as_method("optional(nullopt_t)")]]
    MyOpt(my_null_t);

    [[clang::analyze_as_method("optional(T&&)")]]
    MyOpt(MyT&& val);

    [[clang::analyze_as_method("optional(in_place_t, Args&&...)")]]
    MyOpt(my_inplace_t, Args&&... args);
};
```

## Signature Strings Are Opaque Keys

The signature string is treated as a symbolic disambiguation key, **not** a type
expression to be resolved. The model compares the whole string
(e.g. `"optional(nullopt_t)"`) by equality against its own case labels. It does
not look up the target class, resolve `nullopt_t` (or `iterator`, `T`, …) to a
`QualType`, or inspect the target's real overloads. This matches how the rest of
the model already works: every case fires on a spelled name or an
`analyze_as_method` tag, plus a receiver/result type recognised as optional (via
`AnalyzeAsClassAttr`).

The strings should still *read* like real target signatures
(`"insert(iterator, const T&)"`) — for author clarity, and so a future
validation pass could resolve them against the real target class — but nothing
resolves them today. Resolving types against the target class's actual overloads
(and diagnosing a signature that matches none) is a **later validation phase, not
abandoned** — general overload matching is fully delivered by the opaque-key
mechanism above (one distinct key → one case → one transfer function, for
arbitrarily many overloads) and does not depend on it. See "Backwards
Compatibility" below.

## Matching Strategy

When the model encounters an `analyze_as_method` attribute with a signature string:

1. Retain the full string as the match key (the name/param split is used only for
   Sema well-formedness validation, not for type resolution).
2. Register a `MatchSwitch` case whose matcher fires when a call or construct
   carries that exact signature key *and* the receiver/result type is recognised
   as optional.
3. Attach the transfer function for the intended behaviour to that case.

Because `MatchSwitch` applies the first matching case, an attribute-driven case
that would otherwise be shadowed by a generic case (e.g. the value/conversion
constructor) must be registered **before** it.

This is overload *identification by declared intent*, not overload *resolution*:
the author states which target overload they mean via the string, and the model
routes to the matching case — no candidate ranking or implicit conversions.

## Genericity: not tied to std::optional

The annotation-matching layer must work for **any** target class mapped via
`analyze_as_class` (`std::vector`, `std::unique_ptr`, …), not just `std::optional`.
Two layers, with a clean split:

- **The matcher is target-class-agnostic.** `hasAnalyzeAsMethodName` compares the
  attribute string against the model's query string; it never references
  `std::optional`. Every target class reuses it unchanged.
- **Matching semantics — "accept either":**
  - a *bare* query (`"emplace"`) matches an annotation by its **name part**,
    whether or not the annotation carries a signature;
  - a *signature* query (`"insert(iterator, const T&)"`) matches the **full
    string** exactly.

  This lets annotations self-document with signatures while the model queries by
  bare name for single-outcome operations and by full signature to disambiguate
  multi-outcome overloads — the same rule (see "When a Signature Is Required")
  applied uniformly across all target classes.
- **What stays target-specific:** the model's cases and transfer functions encode
  one class's semantics (for this model, `std::optional`'s
  emplace/reset/nullopt/…). Supporting a genuinely different class (e.g.
  `std::vector`) means adding that class's cases and transfer functions — a
  separate, larger effort — but the annotation-matching layer above is shared.

**Consequence for emplace:** do *not* "fix" it by stripping its signature — that
is an `std::optional`-specific shortcut that fails for classes whose same-name
overloads differ in outcome. Instead keep the signature and make the matcher
accept a bare query against a param'd annotation via the name-part comparison.
That is the generic behaviour; std::optional's `emplace` is just its first
exercise.

## Convenience Macro

A macro can reduce boilerplate and avoid hand-writing signature strings:

```cpp
#define ANALYZE_AS(method, ...) \
  [[clang::analyze_as_method(method "(" #__VA_ARGS__ ")")]]

ANALYZE_AS("insert", iterator, const T&)
// expands to: [[clang::analyze_as_method("insert(iterator, const T&)")]]

ANALYZE_AS("optional", nullopt_t)
// expands to: [[clang::analyze_as_method("optional(nullopt_t)")]]
```

## Backwards Compatibility

A bare method name with no parentheses (e.g. `analyze_as_method("value")`) is
accepted **when the target name is outcome-unambiguous** — every overload the
model recognises under that name produces the same transfer outcome (`value`,
`emplace`, `reset`, `swap`, `has_value`, `value_or`). For these, a bare name and
a full signature are interchangeable.

When a target name is **outcome-ambiguous** — different overloads produce
different outcomes (the constructor set: `optional(nullopt_t)` → empty vs
`optional(T&&)` / `optional(in_place_t, ...)` → engaged) — a full signature is
**required**, and a bare name for that name should be diagnosed. See "When a
Signature Is Required" below.

## When a Signature Is Required

Whether disambiguation is mandatory depends on the *outcomes* of the overloads
under a target name:

- **Same outcome across overloads → optional.** If every overload the model maps
  under a name produces the same analysis outcome, accept **either** the bare
  name or a full signature — they're interchangeable. This is already the
  behaviour of the name-only matcher, which compares only the pre-`(` name.
- **Differing outcomes → signature required.** If the overloads under a name
  produce different outcomes, force disambiguation: each annotation must carry a
  full signature, and a bare name for that name is an error.

This holds identically for constructors. `optional` is outcome-ambiguous (empty
vs engaged), so every custom constructor mapping to it must use a full signature
(`optional(nullopt_t)`, `optional(T&&)`, `optional(in_place_t, Args&&...)`).

**Source of truth.** "Do these overloads differ in outcome?" is answered by the
*model's own case/outcome table* — a finite, known vocabulary of which names map
to which transfer functions — not by resolving the real target class. So this
check needs no `#include` of the target and is independent of the deferred
real-target resolution.

**Where enforced (open).** Because the outcome table lives in the model, Sema
(`isValidAnalyzeAsMethodAttr`) can only validate string *well-formedness*, not
outcome-ambiguity. The "signature required" diagnostic therefore belongs in the
model/check layer, or in a small shared registry that both matching and
validation consult — not in Sema.

## Discarded Alternatives

### Parameter count as a second attribute argument

```cpp
[[clang::analyze_as_method("set", 0)]]   // zero-arg overload
[[clang::analyze_as_method("set", 1)]]   // one-arg overload
```

Discarded because it only works when overloads differ in arity. Same-arity overloads with different parameter types (e.g. `insert(iterator, const T&)` vs `insert(iterator, size_type)`) cannot be distinguished. Since type matching is needed for those cases anyway, parameter count is just a half-measure that delays the same work and results in two code paths instead of one.

### Inferring the target overload from the client method's own signature

```cpp
// Analyser would look at my_insert's params, map MyT→T / MyIter→iterator,
// and find the matching insert overload automatically.
[[clang::analyze_as_method("insert")]]
void my_insert(MyIter pos, const MyT& val);
```

Discarded because it constrains the client method to have parameters that map cleanly to the target method's signature. The whole purpose of annotations is that the custom class can have a completely different interface — different types, different argument order, extra parameters. Tying the two together defeats that goal.

### Separate `analyze_as_constructor` attribute

```cpp
[[clang::analyze_as_constructor("nullopt")]]
MyOpt(my_null_t);
```

Discarded in favour of the unified signature string approach. Constructors are just a special case of overloaded methods — they share a name and need disambiguation by parameter types. A separate attribute would duplicate the same disambiguation logic and add a second concept for users to learn. Using `analyze_as_method("optional(nullopt_t)")` handles constructors with the same mechanism as regular overloads.

### Typestate model (`test_state("engaged/disengaged")`)

```cpp
[[clang::analyze_test_state("engaged")]]
bool HasValue() const;
```

Discarded for the initial proposal. The typestate approach is more general and could support variant-like, resource, and container types with the same vocabulary. However, it adds significant complexity and is not needed for the current use case of mapping custom types to existing standard class interfaces. The simple string-comparison approach can land first; a typestate model could be layered on later if needed.
