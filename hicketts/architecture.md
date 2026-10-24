# Architecture: how the analyze_as_class / analyze_as_method feature flows

Re-onboarding map for the `[[clang::analyze_as_class]]` /
`[[clang::analyze_as_method]]` POC (PR #195054). Read this first after a break —
it traces source → warning across the three subsystems and pins the key
functions. Line numbers are approximate anchors; grep the symbol, don't trust the
number.

See also: `plan.md` (constructor-overload plan), `constructors.md` (background).

---

## 1. End-to-end pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│ SOURCE                                                               │
│                                                                     │
│  hicketts_optional.h                    test_hicketts_optional.cpp   │
│  ┌──────────────────────────────┐      ┌───────────────────────┐    │
│  │ class [[clang::analyze_as_    │      │ HickettsOptional<int> x;│   │
│  │   class("std::optional")]]    │      │ x.unwrap();  // usage   │   │
│  │ HickettsOptional {            │      └───────────────────────┘    │
│  │  [[clang::analyze_as_method(  │              (NB: <optional>       │
│  │     "value")]] unwrap();      │               is NOT included)     │
│  │ };                            │                                    │
│  └──────────────────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────┘
        │
        │  (a) attribute grammar/shape defined here
        v
┌─────────────────────────────────────────────────────────────────────┐
│ ATTRIBUTE DEFINITION — clang/include/clang/Basic/Attr.td             │
│   AnalyzeAsClass   (:924)  StringArgument<"ClassName">               │
│   AnalyzeAsMethod  (:932)  StringArgument<"MethodName">              │
│              │ TableGen generates C++ classes                        │
│              v   AnalyzeAsClassAttr / AnalyzeAsMethodAttr            │
└─────────────────────────────────────────────────────────────────────┘
        │
        v
┌─────────────────────────────────────────────────────────────────────┐
│ PARSE + SEMA  (parse time)  — clang/lib/Sema/SemaDeclAttr.cpp        │
│                                                                     │
│   ProcessDeclAttribute switch (:7663)                               │
│     case AT_AnalyzeAsClass  → handleAnalyzeAsClass   (:6472)         │
│     case AT_AnalyzeAsMethod → handleAnalyzeAsMethod  (:6558)         │
│                    │                                                 │
│                    ├─ validate: isValidAnalyzeAsClassAttr  (:6466)   │
│                    │            isValidAnalyzeAsMethodAttr (:6489)   │
│                    │            (currently ~non-empty only)          │
│                    └─ D->addAttr(AnalyzeAs…Attr(..., Str))           │
│                                                                     │
│   ⚠ std::optional may not exist yet here (not in TU / include order) │
└─────────────────────────────────────────────────────────────────────┘
        │
        v
┌─────────────────────────────────────────────────────────────────────┐
│ AST (the parsed TU)                                                  │
│   CXXRecordDecl  HickettsOptional  ── has AnalyzeAsClassAttr         │
│   CXXMethodDecl  unwrap()          ── has AnalyzeAsMethodAttr("value")│
│   CXXMemberCallExpr  x.unwrap()    ── callee resolved to that decl   │
│                                                                     │
│   Present: custom type + call sites.   Absent: std::optional.       │
└─────────────────────────────────────────────────────────────────────┘
        │
        │  clang-tidy runs bugprone-unchecked-optional-access
        │  → dataflow framework drives the model over each function's CFG
        v
┌─────────────────────────────────────────────────────────────────────┐
│ DATAFLOW MODEL — .../FlowSensitive/Models/UncheckedOptionalAccess…   │
│   (see section 2 — this is the heart)                                │
└─────────────────────────────────────────────────────────────────────┘
        │
        v
┌─────────────────────────────────────────────────────────────────────┐
│ DIAGNOSER — buildDiagnoseMatchSwitch (:1351)                         │
│   at each value-access, is has_value provably true?                  │
│     yes → silent     no → ⚠ "unchecked access to optional value"     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Inside the model (the part you actually work in)

File: `clang/lib/Analysis/FlowSensitive/Models/UncheckedOptionalAccessModel.cpp`

The model is constructed once, then `transfer()` (`:1344`) is called on each CFG
element. Three pieces cooperate:

```
UncheckedOptionalAccessModel ctor (:1325)
│
├─ (A) TYPE RECOGNITION — "is this type an optional?"
│      hasOptionalClassName (:63)
│         ├─ hardcoded names: "optional" in std/absl, "Optional"      ← the model's
│         │  in base/folly, … (:67-97)                                  hand-written
│         └─ OR  RD.hasAttr<AnalyzeAsClassAttr>()  (:99)  ← your hook    knowledge of
│      getOptionalBaseClass (:105) walks base classes                   std::optional
│
├─ (B) SYNTHETIC FIELDS — setSyntheticFieldCallback (:1330)
│      for any recognised optional type, attach:
│         "has_value" : bool
│         "value"     : valueTypeFromOptionalDecl (:474)
│                       = template arg [0]   ← the shortcut Valentyn's
│                                              point would replace
│
└─ (C) TRANSFER MATCH SWITCH — buildTransferMatchSwitch (~:1000)
       an ordered list of  CaseOfCFGStmt<NodeKind>(matcher, transferFn)
       FIRST match wins  → ordering matters (nullopt before value!)

   ┌── matcher ──────────────────────────────┐   ┌── transfer fn ─────┐
   │ isOptionalNulloptConstructor (:289)      │→  │ setHasValue(false) │
   │   arg0 is nullopt_t  OR                   │   └────────────────────┘
   │   hasAnalyzeAsMethodName("optional(       │
   │     std::nullopt_t)")  (:239)            │
   ├──────────────────────────────────────────┤   ┌────────────────────┐
   │ isOptionalInPlaceConstructor (:297)      │→  │ setHasValue(true)  │
   │ isOptionalValueOrConversionCtor (:302)   │→  │ setHasValue(true)  │
   ├──────────────────────────────────────────┤   ┌────────────────────┐
   │ value()/unwrap() call:                   │→  │ read has_value;    │
   │   hasName("value") OR                     │   │ if not-true here → │
   │   hasAnalyzeAsMethodName("value") (:239) │   │ flag for diagnoser │
   └──────────────────────────────────────────┘   └────────────────────┘
```

The one function to re-read first is **`hasAnalyzeAsMethodName` (`:239`)** — it's
the entire bridge between your attribute and the model:

```
if query contains '('  →  AttrValue == query          (full-string key)
else                   →  AttrValue.split('(').first == query   (name only)
```

That's the "opaque key": the string is compared, never resolved.

---

## 3. The conceptual overlay — the "two worlds"

This is the mental model that untangles most of the confusion.

```
        THE CUSTOM TYPE                    STD::OPTIONAL (the reference)
        ───────────────                    ─────────────────────────────
  HickettsOptional, x.unwrap()        the model's IDEA of std::optional

  WHERE: real decls in the AST        WHERE: hardcoded in the model source
         (header is in the TU)               (name lists + ctor cases)

  HAVE: name AND full signature       HAVE: only what a human typed in
        (params, return type, …)             (:63, :275, :283, :289-308)

  ✓ always present                     ✗ real class usually NOT in the TU

  ── the attribute STRING is the bridge between them ──
     analyze_as_method("value")  = "treat this custom method
                                     like std::optional's value"
     matched by string equality to a model case label —
     no lookup, no type resolution
```

Takeaways that keep mattering:
- **`hasAnalyzeAsMethodName` (`:239`)** is where your attribute meets the model.
- **Recognition happens two ways** — hardcoded names *or* your `AnalyzeAsClassAttr`
  (`:99`); same idea for methods (`hasName(...)` *or* `hasAnalyzeAsMethodName`).
- **Ordering in the match switch is load-bearing** — nullopt cases before the
  generic value case, or the value case eats the nullopt tag.
- **`valueTypeFromOptionalDecl` (`:474`)** is the `template-arg-[0]` shortcut that
  Valentyn's return-type point would have to replace for general targets.

---

## 4. Design axis: how much should the tool *verify* vs *trust*?

The recurring design question, framed as levels of validation. Key fact: the
**custom** type is always in the TU; **std::optional** may never be. So:

- **Level 0 (current MVP):** the annotation string is an opaque key. No validation;
  custom param types never checked. Trusts the annotator completely.
- **Level 1 — validate the annotation *string*** against a hardcoded table of
  known std::optional operations. Catches typos. **Header-free; can run in Sema
  at parse time** (revive `isValidAnalyzeAsMethodAttr`, :6489, to check a real
  operation list instead of balancing parens).
- **Level 2 — also validate the custom method's *arity/shape*.** The custom
  method's real signature *is* available in Sema (`handleAnalyzeAsMethod` gets the
  method decl `D`, :6558). Compare its arity against the hardcoded expected shape.
  **Still header-free, still parse-time.**
- **Level 3 — validate custom param *types* correspond to std's.** Breaks down:
  the custom tag (`nothing_t`) deliberately differs in name from `std::nullopt_t`,
  so name comparison would reject valid code. Needs tag registration or structural
  cues. **This is the rabbit hole — and it's independent of whether std is loaded.**

Conclusion: **Levels 1–2 are the sweet spot** and dissolve the "validation forces
the header" dilemma (validate against the hardcoded table + the custom decl, both
present — you never needed the real std::optional). Level 3 is high-friction,
low-value.

### The `#include <optional>` blocker (why Plan A stalled)

Plan A = validate against the **real** std::optional signatures. That is the *only*
design that needs the real class in the TU. And the blocker is **not** about
timing:

- Fundamental (layer-independent): std::optional isn't guaranteed to be in the TU
  at all — nothing forces a file using the custom type to `#include <optional>`.
  True at parse time *and* model time.
- Parse-time-only extra wrinkle: even if the TU includes `<optional>`, at the
  moment Sema handles the attribute on the custom class, it may not have been seen
  yet (include ordering). Model time doesn't have this second problem.

So the header dependency is fundamental to validating-against-std, not an artifact
of *when* the check runs — moving to the model layer does not fix it.

### Alternatives in play

- **This PR (MVP):** argument strings as opaque keys in `analyze_as_method`. No
  signature mapping to the template class.
- **BaLiKfromUA's POC:** bare `analyze_as_method("optional")`, disambiguate custom
  ctors by **arity-correspondence** to std::optional's ctor set. Open question:
  does he match against the *real* std decl (needs header) or a *hardcoded table*
  (header-free)? Either way, two 1-arg overloads with different outcomes
  (`optional(nullopt_t)` vs `optional(T&&)`) collide on arity and still need a
  type signal — where tag-registration / an explicit outcome tag plugs the gap.
- **Return types (Valentyn):** orthogonal to arity matching. C++ forbids
  overloading on return type alone, so it's only ever a tie-breaker — but for
  *general* targets the contained type must come from the unwrap method's return
  type (replacing `valueTypeFromOptionalDecl`'s template-arg-[0] shortcut).
```
