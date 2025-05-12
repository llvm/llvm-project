```{title} clang-tidy - performance-use-std-move
```

# performance-use-std-move

Finds nontrivial copy construction and copy assignment that could select a move
operation instead, when the current value of an automatic variable has no
subsequent use on the analyzed paths. The check works with user-defined types;
it does not recognize containers or reset functions by their names.

```cpp
void construct(std::string source) {
  inspect(source.size());
  std::string target(source); // suggests std::move(source)
}

void assign(std::vector<int>& target, std::vector<int> source) {
  target = source; // suggests std::move(source)
}
```

The diagnostic is `'source' could be moved here`. A diagnostic is a performance
recommendation, not a proof that the transformation preserves every observable
behavior of an arbitrary class.

## Covered cases

The following table describes coverage after the eligibility, last-use, and
alias checks below. **Advisory** means a diagnostic without an automatic fix.

| Copy site | Behavior |
| --- | --- |
| Direct initialization of a local, `T target(source)` | Fix when the source and destination have the same lexical scope |
| Copy initialization, `T target = source` | Fix when the applicable move constructor is nonexplicit |
| List initialization, `T target{source}` | Fix unless initializer-list overloads could change constructor selection |
| Copy assignment, `target = source` | Fix for an applicable move assignment and an owned source |
| By-value function or constructor argument | Advisory; changing the argument can change overload resolution |
| Explicit temporary, `accept(T(source))` | Advisory |
| Heap construction, `new T(source)` | Advisory |
| Aggregate element, `Aggregate{source}` | Advisory |
| A return expression containing a genuine copy | Advisory; implicit moves and elidable copies are excluded |
| Copy into a shorter-lived local scope | Advisory |
| Named rvalue-reference source | Advisory; the caller's object may be affected |
| Copy capture, `[source]`, `[=]`, or `[copy = source]` | Advisory; no capture rewriting |
| Macro expansion | Advisory when analyzable; the macro spelling is not rewritten |
| Concrete function-template instantiation | Advisory, deduplicated at the copy spelling |
| A lambda's own parameters and locals | Analyzed in the lambda's callable context |

Aliases of types and parentheses do not prevent detection:

```cpp
using Text = std::string;
void aliases(Text source) {
  Text target((source)); // becomes ((std::move(source)))
}
```

A trivial move can replace a nontrivial copy. The check examines the selected
copy operation instead of treating a nontrivial destructor as evidence of an
expensive copy. Nontriviality is a cost heuristic: the check does not measure
runtime, inspect allocator costs, or guarantee that moving is faster.

A move must accept a mutable, nonvolatile rvalue reference and be public,
nondeleted, unconstrained, and unambiguously applicable within the supported
special-member cases. The source and destination types
must agree after removing qualifiers. Constructor explicitness and the move
assignment's receiver qualifiers are taken into account. A move that merely
exists but cannot be selected is not sufficient:

```cpp
struct Object {
  Object(const Object&);
  explicit Object(Object&&);
};
void explicitMove(Object source) {
  Object target(source); // direct initialization can use the explicit move
}
void ineffectiveMove(Object source) {
  Object target = source; // std::move would still select the copy: no diagnostic
}
```

## Last uses, aliases, and restored values

The analysis uses a control-flow graph and expression sequencing. It checks
branches separately, considers exception paths, and does not equate textual
source order with execution order.

```cpp
void branches(bool condition, std::string source) {
  if (condition)
    consume(source); // advisory: last use on this path
  else
    consume(source); // advisory: last use on the alternative path
}

void laterUse(std::string source) {
  consume(source);   // no diagnostic
  inspect(source);  // source is still used
}
```

Local reference aliases, including transitive aliases, are tracked. An alias
that is used only before the copy need not prevent the recommendation. Passing
an alias to an unknown reference-taking function may retain access and prevents
a recommendation even if that call precedes the copy.

```cpp
void localAlias(Value source) {
  const Value& first = source;
  const Value& second = first;
  inspect(second.value);
  Value target(source); // eligible
}

void retainedAlias(Value source) {
  retain(source);       // retain(const Value&) may retain a reference
  Value target(source); // no diagnostic
}
```

Pointer aliases and retained references or pointers returned by member functions
are handled conservatively. Taking an address for a call with an applicable
`[[clang::noescape]]` parameter contract on an ordinary function or member call
need not be a permanent escape. Constructor arguments are conservatively treated
as possible escapes without using that annotation.
The check trusts that annotation; it does not verify the callee's implementation.

Objects created within a loop are distinguished from values reused across
iterations:

```cpp
void fresh(int count) {
  for (int i = 0; i < count; ++i) {
    Value source;
    Value target(source); // eligible: a new source exists each iteration
  }
}

void repeated(Value source, int count) {
  for (int i = 0; i < count; ++i)
    consume(source); // no diagnostic: later iterations reuse the same value
}
```

Under conventional value semantics, an independent, nonthrowing copy or move
assignment restores a value. A nonthrowing member function with
`[[clang::reinitializes]]` supplies an explicit restoration contract. Ordinary
mutating calls, compound assignments, and arbitrary non-const reference
parameters do not supply that contract. Throwing restorations are conservatively
not used to prove that the old value is dead.

```cpp
void restoration(Value source) {
  consume(source);          // advisory: this value is not used again
  source = Value{};         // independent, nonthrowing value assignment
  inspect(source.value);   // uses the restored value
}

void incompleteRestoration(bool condition, Value source) {
  consume(source);          // no diagnostic
  if (condition)
    source = Value{};
  inspect(source.value);   // not restored on every path
}
```

Unevaluated expressions, such as an ordinary `sizeof(source)`, do not count as
runtime reads. Expressions whose evaluation order is unknown are rejected;
ordered expressions are not rejected merely because the source appears twice.

```cpp
consume((inspect(source.value), source)); // advisory: comma orders the read
consumeTwo(source, source);               // no diagnostic: argument order varies
```

## Automatic fixes and their guarantees

Construction fixes are restricted to owned local variables or value parameters
and directly initialized automatic destinations in the same lexical scope.
Assignment fixes are withheld for rvalue-reference sources, which can alias a
caller-visible target. Known self-assignment is not diagnosed, including casts
and local aliases. An assignment receiver depending on the source is also
excluded because it could designate the same object. Fixes preserve
the source expression's spelling, validate the editable file range, and insert
`<utility>` when needed.

Scopes introduced by control statements also matter:

```cpp
void shorterLoopLifetime(Value source) {
  for (Value target(source); again(); ) // advisory: target dies when the loop ends
    break;
}
```

Function arguments, captures, macros, and function-template instantiations are
not automatically rewritten. In particular, a movable observed template
instantiation does not establish that rewriting the template is valid for all
future instantiations.

If a potentially throwing move replaces a nonthrowing copy, it can introduce
an exceptional path absent from the original graph. A use of the source in an
enclosing handler prevents the recommendation; otherwise the diagnostic is
advisory.

```cpp
void newExceptionPath(Value source) {
  try {
    Value target(source); // no recommendation if copy is noexcept, move can throw
  } catch (...) {
    inspect(source.value);
  }
}
```

These restrictions reduce mechanical rewriting risks. They do **not** establish
arbitrary semantic equivalence. The check assumes conventional value behavior
for copy, move, and independent assignment. It cannot generally determine:

- Whether constructors, assignments, or destructors have intentional observable
  side effects, or whether different exception behavior is acceptable.
- Whether the source must retain ownership until a particular point. Moving a
  shared owner of a lock or scope guard can release a resource earlier even
  without another direct source reference.
- Whether an arbitrary member function retains `this` or an internal reference
  through hidden state. Explicit retained reference results are tracked, but
  interprocedural escape analysis is not performed.
- Whether a move preserves aliases into the object's internal storage.
- Whether a recommendation improves performance on the actual workload.

Same lexical scope, an available move, and `noexcept` are useful restrictions,
not proofs of all these properties. Review recommendations involving
lifetime-sensitive or unusual value types; exclude them with `AllowedTypes` or
normal `NOLINT` suppression.

## Exclusions and remaining gaps

No recommendation is made for const or volatile sources, lvalue-reference
source variables, global/static/thread-local/external source storage, trivial
copies, unavailable moves, existing moves, or elidable construction. A copy
through an lvalue-reference alias is excluded even when the alias refers to a
local; aliases are tracked to assess copies from the original variable.

The following cases require additional analysis and are not supported:

| Gap | Example or required analysis |
| --- | --- |
| Forwarding APIs and reference-taking functions that copy internally | `container.emplace_back(source)`; needs generic interprocedural copy/escape summaries |
| Object fields and array or container elements as sources | `consume(owner.field)`, `consume(array[0])`; needs access-path and owner-lifetime analysis |
| Arbitrary pointer-alias liveness | `T* alias = &source`; conservatively excluded even if all pointer uses precede the copy |
| Unannotated reset methods and throwing restorations | Method names and mutation alone do not establish restoration |
| RHS-dependent restoration | `source = transform(source)`; not treated as an independent restoration |
| Restoration through aliases or computed receivers | `alias = T{}`; needs a proof of whole-object identity, not just an alias into a base subobject |
| Complex move overloads and constrained special members | Requires counterfactual overload resolution; a private move is excluded even in an otherwise authorized access context |
| Additional explicit copy-constructor arguments, variadic moves, and explicit-object move assignment | Need argument and overload modeling beyond ordinary special-member calls |
| General path feasibility | Incompatible branch conditions can cause conservative missed recommendations |
| Constructor member initializers outside the analyzed body | Need additional CFG and construction-context support |
| Coroutines | Suspension, frame storage, escaping references, and destruction need lifetime modeling |
| Template-wide and macro-wide fixes | Require validity beyond observed instantiations or expansions |
| Uninstantiated dependent code and a generic `std::forward` recommendation | Need instantiated copy and move facts; a concrete rvalue-reference instantiation can receive an advisory `std::move` diagnostic |
| Language sequencing not represented by the supported expression analysis | Conservatively missed, particularly overloaded operator expressions |

CFG construction or an expression-to-block mapping failure causes the check to
skip the recommendation. Invalid and incomplete code may therefore have less
coverage. These gaps are deliberate conservative misses; the check has no mode
that bypasses alias or last-use checks to generate automatic fixes.

## Options

### AllowedTypes

A semicolon-separated list of regular expressions matching types that should
not be diagnosed. The default is an empty list. Patterns containing `::` are
matched against a qualified name; other patterns match the unqualified name.
Aliases resolve to the underlying record name. The option applies to both
construction and assignment.

For example, `ScopeGuard;::application::.*Lease` excludes an unqualified
`ScopeGuard` and matching application lease types. The check has no built-in
list of non-owning views or scope guards.

### IncludeStyle

A string specifying the include style, `llvm` or `google`. The default is `llvm`.
