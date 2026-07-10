# Plan: Constructor Overload Disambiguation via Signature Strings (POC)

## Why start with constructors

Signature disambiguation only *earns its keep* when two overloads of the same
name need **different** transfer functions. For `emplace`, both overloads set
`has_value = true`, so disambiguating them changes nothing observable — a bad
place to start because you can't tell whether it works.

Constructors are the opposite: `optional(nullopt_t)` → `has_value = false` but
`optional(T&&)` / `optional(in_place_t, ...)` → `true`. Same name, different
behaviour. Pick the wrong overload and the analysis is wrong — so there's a
clear pass/fail signal.

Constructors are the *first* slice, not the whole feature. General overload
matching — disambiguating arbitrary overloaded methods on any target class —
stays the goal (see `constructors.md`). Regular overloaded methods are the very
next phase (Step 6), reusing the *same* signature-key mechanism; only the
transfer functions and the matched node kind (member call vs construct) differ.

## How the model actually works (design decision this resolves)

The model never looks up `std::optional`'s real declaration. Each case in the
`MatchSwitch` fires when a call/construct is spelled a known name **or** carries
an `analyze_as_method("X")` tag, *and* the receiver/result type is recognised as
optional (via `AnalyzeAsClassAttr`, see `hasOptionalClassName`). The attribute
string is a **symbolic disambiguation key** compared against the model's case
labels — not a query resolved against the target class's overloads.

Consequence: the signature string (`"optional(nullopt_t)"`) is matched by string
equality to a case label. We do **not** resolve `nullopt_t` to a real QualType or
compare against `std::optional`'s actual constructor signatures. (This supersedes
the earlier "compare `QualType::getAsString()` against the target overload"
idea in `constructors.md` — that would be a foreign kind of lookup the model has
never needed, and it's blocked anyway because the test header doesn't
`#include <optional>`.)

## Bare name vs required signature (design rule)

Whether a full signature is *required* depends on the target name's outcomes:

- **Same outcome across overloads → bare name or full signature both accepted.**
  Names like `emplace`, `reset`, `swap`, `value`, `has_value`, `value_or` map
  every overload to one outcome. The existing name-only matcher
  (`hasAnalyzeAsMethodName`, which compares only the pre-`(` name) already accepts
  either spelling — this half needs no new work.
- **Differing outcomes → full signature required, bare name is an error.** The
  constructor name `optional` is the case: `optional(nullopt_t)` → empty vs
  `optional(T&&)` / `optional(in_place_t, ...)` → engaged. Each custom
  constructor must carry a full signature, and a bare `"optional"` should be
  diagnosed.

Source of truth: ambiguity is decided by the *model's* case/outcome table (a
finite, known vocabulary), not by resolving the real target class — so it needs
no `#include` and is independent of the deferred real-target resolution.

Practically, each supported target name is either *name-matched* (unambiguous) or
*signature-matched* (ambiguous). Reifying that as a small registry
(name → [(signature, outcome)]) lets both the matcher and the "signature
required" check read from one place. Open: that diagnostic can't live in Sema
(which sees only the string, not outcomes) — it belongs in the model/check layer.

## Current state of constructor handling

The model already has three constructor cases (UncheckedOptionalAccessModel.cpp
`:288`–`:306`), but they match structurally by the **argument's real type name**,
not by attribute:

- `isOptionalNulloptConstructor` — arg type is literally `std::nullopt_t` (or
  absl/base/folly/bsl variants) → `has_value = false`
- `isOptionalInPlaceConstructor` — arg type is `in_place_t` → `true`
- `isOptionalValueOrConversionConstructor` — any other single-arg ctor → `true`

**The bug this POC fixes:** a custom optional whose "empty" constructor takes a
differently-named tag type (`mylib::nothing_t`) is not recognised as the nullopt
constructor, so it falls through to the value/conversion case and is wrongly
marked engaged. Today this is a false negative:

```cpp
mylib::HickettsOptional<int> x{mylib::nothing};
x.unwrap();   // should warn (empty) — currently does NOT
```

## Goal (testable milestone)

Tag the custom constructors and have the model route them to the correct
transfer function by signature:

```cpp
[[clang::analyze_as_method("optional(nullopt_t)")]]      HickettsOptional(nothing_t);   // -> false
[[clang::analyze_as_method("optional(T&&)")]]            HickettsOptional(T&&);          // -> true
[[clang::analyze_as_method("optional(in_place_t, Args&&...)")]] HickettsOptional(my_inplace_t, Args&&...); // -> true
```

Success = `x{mylib::nothing}; x.unwrap();` now **warns**, while
`x{42}; x.unwrap();` stays quiet.

## Steps

1. **Sema validation** — already done (well-formedness of the signature string).
   No further work needed for the POC.

2. **Write the failing test first.** Add the tagged constructor(s) to
   `hicketts_optional.h` and expectations to `test_hicketts_optional.cpp`, then
   run clang-tidy and *watch the nullopt case fail to warn*. This grounds the
   target and confirms the value/conversion case is what's (wrongly) firing.

3. **Signature-aware matcher.** The existing `hasAnalyzeAsMethodName` (`:239`)
   does `AttrValue.split('(').first`, so it strips params and cannot tell
   `optional(nullopt_t)` from `optional(T&&)`. Add a matcher that compares the
   **full** attribute string to a given signature label (e.g.
   `hasAnalyzeAsSignature("optional(nullopt_t)")`).

4. **Attribute-driven constructor cases.** Mirror the three structural cases,
   but gate them on the signature key instead of the arg's type name:
   - `"optional(nullopt_t)"` → `setHasValue(false)`
   - `"optional(T&&)"` and `"optional(in_place_t, Args&&...)"` → `setHasValue(true)`
   - a bare `"optional"` tag (no signature) → diagnose "signature required" per
     the design rule above; do not silently route it to a constructor case

   **Ordering matters:** `MatchSwitch` applies the first matching case, so the
   attribute-driven nullopt case must be registered *before*
   `isOptionalValueOrConversionConstructor` — otherwise the generic value case
   greedily matches first and the nullopt tag never gets a chance.

5. **Verify.** Build clang-tidy, run it on the fixture. The nullopt-constructed
   `unwrap()` should now warn; the value-constructed one should not.

6. **Phase 2 — regular overloaded methods.** Apply the same full-signature
   matcher to the member-call path (`CXXMemberCallExpr`), so overloaded *methods*
   are disambiguated exactly like constructors — this is the general
   overload-matching capability, not a throwaway. Concretely, tag the two
   `emplace` overloads with distinct keys (`"emplace(Args&&...)"` vs
   `"emplace(initializer_list<U>, Args&&...)"`) and confirm both route to their
   case.

   Honesty about observability: within `std::optional` both `emplace` overloads
   set `has_value = true`, so a *behavioural* difference between two same-named
   method overloads only shows up with a richer target class (e.g.
   `vector::insert` variants, or a custom `set()`-resets vs `set(T)`-assigns
   pair). The matching *mechanism* is identical and is proven here; the richer
   behavioural cases land when `analyze_as_class` is generalised beyond
   `std::optional`.

## Deferred (future phase, not abandoned)

Resolving/validating a signature string against the target class's **real**
overloads — looking up `std::optional`/`std::vector` in the AST, pulling its
actual `QualType`s, and diagnosing a signature that names no real overload. This
is purely a *validation/safety* layer; overload matching itself does not need it.
It requires the target class to be present in the translation unit (an
`#include`) plus an AST lookup the model doesn't currently do, so it's a later
phase — the matching feature is complete without it.

**TODO (spike, after the POC): scope the effort for real param validation.**
Investigate — don't implement — how much work real parameter validation would be,
and produce a short write-up with a rough estimate. Questions to answer:
- How to obtain the target decl: require an `#include` of the target header, or
  look it up by qualified name via `ASTContext` / `Sema` lookup?
- Where it runs: Sema attribute handling vs the model/check layer.
- Type-spelling mismatches: canonicalisation, template params (`T`, `Args&&...`),
  sugar/aliases — how strict must the comparison be, and via what API
  (`QualType::getCanonicalType`, `getAsString` with a `PrintingPolicy`, …)?
- What diagnostics to emit (signature names no real overload / ambiguous) and
  their wording.
Deliverable: a paragraph or two estimating scope, not code.

## Files to modify

1. `clang/lib/Analysis/FlowSensitive/Models/UncheckedOptionalAccessModel.cpp`
   — new full-signature matcher + attribute-driven constructor cases (ordered
   before the generic value/conversion case)
2. `hicketts/hicketts_optional.h` — tagged constructors + `my_inplace_t` tag type
3. `hicketts/test_hicketts_optional.cpp` — nullopt-vs-value construction cases
4. (Later) unit + clang-tidy integration tests in the real tree
5. Sema — already done

## Verification

1. Build: `cmake --build build-llvm --target clang-tidy`
2. Fast loop on the fixture (from `hicketts/`):
   ```
   clang-tidy -checks='bugprone-unchecked-optional-access' \
     test_hicketts_optional.cpp -- -I . -Wno-undefined-inline
   ```
3. Later: unit test binary for `UncheckedOptionalAccessModelTest` and `llvm-lit`
   on the `unchecked-optional-access` integration test.
