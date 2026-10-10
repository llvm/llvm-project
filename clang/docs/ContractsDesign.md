# Introduction

This is a high level design documentation for Contracts implementation in Clang.
This summarized experiences from several implementors in seperate branches.

This is also tend to be some developer policy/guide of contracts for how should clang
developers continue working on this.

## High Level Developing Style

We plan to implement contracts in different small PRs. We don't want to implement
contracts in a single PR. Smaller PR helps reviewers to understand the patch and give
higher quality reviewers. Samller PR is also easier for testers to test and find the
problems.

The implementation contracts is marked as experimental before we announce it is 
implemented completely.

### ABI

We'd like to keep ABI unstable during the experimental process. We all understand
ABI's importance. But getting consensus for ABI needs a long time. We prefer to
move faster instead of waiting the consensus for ABI consensus.

For GCC/libstdc++ interoperability, we need to identify the ABI revision
which a toolchain targets. We shouldn't assume that one experimental GCC ABI
will remain unchanged.

### Driver flags

Before we think the support of contracts are minimal usable, users are unable
to get access to contracts unless users pass `-Xclang` explicitly. 

## AST structure

### Contract Stmt

```C++
class Stmt {

class ContractStmtBitfields {
    unsigned Invalid : 1;
};
};

class ContractStmtBase : public Stmt {
  Expr *Predicate = nullptr;
  SourceLocation KeywordLoc, LParenLoc, RParenLoc;
  // llvm::ArrayRef<const Attr *> Attributes; // The design allows contracts to have attributes but none
                                              // contracts attribute are presenting right now.
};

class ContractAssertStmt : public ContractStmtBase {};

class ContractPreStmt : public ContractStmtBase {};

class ContractPostStmt : public ContractStmtBase {
  ResultNameDecl *ResultName = nullptr;
};
```

Discussion:

We though to use a simple struct to represent `pre` and `post` contracts. But we decided to put all of
them into Stmt system. The major reason is that now the users can reuse components like 
`TraverseStmt(...)` more easily.

For attributes, as we don't have any attributes for contracts yet, we can save this on first implemenation.

We split them into 3 Stmt kinds so we can save the bits for ResultName for other contracts types.

For `ResultNameDecl`, see following sections.

### FunctionDecl

```C++
class FunctionContractSequence final
    : private llvm::TrailingObjects<FunctionContractSequence, ContractStmtBase *> {
  friend llvm::TrailingObjects<FunctionContractSequence, ContractStmtBase *>;
  unsigned NumContracts;

  explicit FunctionContractSequence(unsigned N) : NumContracts(N) {}

public:
  static FunctionContractSequence *Create(
      ASTContext &Ctx, llvm::ArrayRef<ContractStmtBase *> Contracts);

  llvm::ArrayRef<ContractStmtBase *> contracts() const {
    return {getTrailingObjects<ContractStmtBase *>(), NumContracts};
  }
};

class FunctionDecl : /*...*/ {
  /*...*/
private:
  // The actual contracts of the function.
  // Even if it is not explicitly written.
  FunctionContractSequence *Contracts = nullptr;

public:
  bool hasContracts() const {
    return Contracts;
  }

  void setContracts(FunctionContractSequence *ContractsSeq) {
    Contracts = ContractsSeq;
  }

  const FunctionContractSequence *getContracts() const {
    return Contracts;
  }
};
```

Discussion:

We need to use a single sequence for the contracts of a function. e.g.,

```C++
int f(int x)
  pre(x > 0)
  post(r: r > 0);

int f(int x)
  post(r: r > 0)
  pre(x > 0); // Illegal
```

So that we can't use a sequence for `pre` and another sequence for `post`.

Another problem is that we allow redeclarations without contracts, or
with semantically equal contracts. So

```C++
int func(int lhs, int rhs) pre(lhs != rhs); // FD1
int func(int lhs1, int rhs1) { return lhs1 == rhs1; } // FD2
```

we should think the function declaration `int func(int lhs1, int rhs1)`
(FD2) have the same contracts as the first one.
Then it is a problem that how should we store/access the contracts for
`FD2`.

Usually, as this is related to redeclarations, we might think we can
get it by iterating redecls. e.g.,

```C++
for (auto *RD : this->redecls())
  if (auto *ContractSeq = RD->getContracts())
    return ContractSeq;
```

But if we follow this, the contracts we get for `FD2` will be the contracts
of `FD1`, `pre(lhs != rhs)`. While this is semantically correct,
`FD2` doesn't have `lhs` or `rhs`. It is not only they have
different name but they are actually different AST nodes.
So it will be problematic for code generation or static analysis
if we simply store the contracts to the declaration
which the contracts are written.

The simplest solution is that Sema to put the corresponding
contracts on every function redeclaration.

This means, **Sema is responsible to construct a new contract
sequence when it saw a redeclaration if the first function has
contracts, no matter if the redeclaration has contracts written
or not**.

#### Possible Later Changes

If needed, we can add `IsWritten` information to `FunctionContractSequence`
to indicate if this sequence is written in source or not.
This may be helpful for toolings.

But in the first version, we don't have to do that.

### ResultNameDecl

A post contract can introduce a name to represent the return value of the
function. e.g.,

```C++
int f(int x) post(result: result > x);
```

We introduce a new `ValueDecl` for this name.

```C++
class ResultNameDecl final : public ValueDecl {
  ResultNameDecl(DeclContext *DC, SourceLocation Loc,
                 IdentifierInfo *Name, QualType T);

public:
  static ResultNameDecl *Create(ASTContext &Ctx, FunctionDecl *FD,
                                SourceLocation Loc, IdentifierInfo *Name,
                                QualType T);
};
```

We don't use `VarDecl` here because the result name doesn't introduce a new
variable. It doesn't have its own storage, initializer or lifetime. During
code generation, references to the `ResultNameDecl` are bound to the return
value of the function. Sema rejects a result name on a function returning
`void`.

## Parser

When implementing parser, an important detail is that we need to
deley parsing for member functions.

```C++
struct S {
  bool f() const pre(value > 0 && valid());

  bool valid() const;
  int value;
};
```

A function contract specifier in a member declaration is a complete-class
context. So `value` and `valid` are both valid names in the predicate, even
though they are declared after `f`.

When Parser first sees the `pre` specifier, the class is not complete yet. If
we parse the predicate immediately, name lookup can't find these members. It
may also see an incomplete overload set when the predicate calls a member
function. This will reject valid programs.

So for contracts on a member function declared inside a class, Parser only
collects the tokens of the predicate and keeps their source locations. After
the class is completely parsed, Parser re-enters the class and function scope,
puts the parameters and `this` back into scope, and parses the cached tokens
through the normal Sema path.

Free functions and out-of-class member declarations don't have this problem
and can parse their predicates immediately. `contract_assert` is parsed in a
function body and doesn't need this delayed parsing either.

We should reuse Clang's existing late-parsed method declaration mechanism,
which is already used for other complete-class contexts. We shouldn't add a
second delayed parsing mechanism only for Contracts.

## Construct Contracts

We should reuse normal expression analysis for contract predicates, including
the contextual conversion to `bool` and full-expression cleanup handling.
We still check the predicate when the evaluation semantic is `ignore`.

In the above FunctionDecl section, we've mentioned that, Sema is reponsible
to rebuild the contracts for every redeclaration to make sure the relationships
between parameters and contracts are correct.

Except this, there are several other details which need special care.

### contract context

We need a contract context in Sema.We maintain a stack of contract contexts in Sema.
Before analyzing each pre, post or contract_assert predicate, we push a new context.
When the analysis finishes, we pop it and restore the previous context. It records
the scope where the predicate starts and the result name, if there is one. The result
name is only visible in its own postcondition. The context needs to support nesting,
since a lambda in a predicate can contain another `contract_assert`.

An important use of this context is implicit const qualification. e.g.,

```C++
bool check(int &);
bool check(const int &);

void f(int x) pre(check(x)) { // Calls check(const int &).
  check(x);                   // Calls check(int &).
}
```

Sema should apply implicit const qualification when building references in
the predicate, before overload resolution. We shouldn't change the type of
`ParmVarDecl`, since that would also change the function body. Applying const
after the predicate is built is too late: we may already have selected the
wrong overload.

Another point we need to care is, when deciding if we're still in the contracts,
we need to look at the decl contexts too instead of looking at the contract context
only. Otherwise, we may meet problems if the predicate triggers some instantiations.
e.g.,

```C++
template <typename T>
auto positive_after_increment(T value) {
  ++value; // legal
  return value > 0;
}

void f(int x) pre(positive_after_increment(x));
```

If we decide if we're in contracts by contract context only, we may think `++value`
in the instantiation as incorrect due to it didn't meet the bar of implicit
constness.

### Lambdas

We need to extend lambda capture analysis to handle contract scopes. e.g.,

```C++
void f(int x) pre([x]() mutable { return ++x > 0; }()); // OK.
void g(int x) pre([&x] { return ++x > 0; }());          // Illegal
int h() post(r: [r] { return r > 0; }());              // OK
```

In `f`, `x` refers to a copy introduced inside the predicate. We shouldn't
reject as `x` changes. In `g`, it still refers to the parameter outside the
predicate, so the reference is implicitly const. We need to follow the capture
chain for nested lambdas to make this distinction. Simply making every expression
const while a contract is being checked will reject valid programs.

Another problem is that we parse a function's contracts before entering its
body. e.g.,

```C++
int check(int x)
  pre([=] {
    return [=] { return x > 0; }();
  }())
  post(r: [r] { return r > 0; }());
```

Here `check` is only a declaration. Both lambdas in the precondition need to
capture `x`, and the lambda in the postcondition needs to capture `r`.

When parsing the precondition, the `FunctionDecl` may not even exist yet,
and `x` may not have its final `DeclContext`. If capture analysis assumes
that a parameter without a function `DeclContext` doesn't need capture, it
will skip the captures of `x`. If it assumes there is already an enclosing
function body, walking out of the nested lambdas may also leave the
declaration context and function scope stack inconsistent.

So we should use the recorded contract scope boundary when analyzing these
captures, and keep the declaration context and function scope stack consistent.
Each intervening lambda still needs its own capture. For `[r]`, capture
analysis also needs to accept `ResultNameDecl` as a capturable local entity,
instead of assuming that every captured name is a `VarDecl`.

The standard allows a capture-default or simple-capture when the enclosing
scope is a contract-assertion scope, even without a function body.
[expr.prim.lambda.capture, paragraph 3.3](https://eel.is/c++draft/expr.prim.lambda.capture#3.3)
states:

> it appears within a contract assertion and its innermost enclosing scope is
> the corresponding contract-assertion scope ([basic.scope.contract]).

### Template Instantiation

For templates, we also need to instantiate contracts on odr-use, even when
the function body isn't available. So we shouldn't tie contract instantiation
only to function body instantiation. e.g.,

```C++
template <typename T>
void f(T value) pre(value.size() > 0);

void use_f() {
  f(1); // Illegal: int has no member named size.
}
```

The call odr-uses `f<int>`, so Sema needs to instantiate its predicate. There
is no function body available here, but we still need to diagnose the invalid
member access. If we only instantiate contracts when instantiating the body,
we will miss this error.

## Constract Semantics Checking

We should share the function-level checks between normal function declarations,
late-parsed member contracts, lambda call operators and template instantiations.
Putting all checks in `ActOnFunctionDeclarator` is not enough, since lambda
call operators are built through a different path.

We run each check when the information it needs is available. Checks depending
on the return type run after deduction, while checks depending on the function
body run after the body is analyzed. A failed check should mark the contract
invalid so later checks don't diagnose the same broken predicate again.

### Postcondition parameters

Sema needs to check the declared type of a non-reference parameter odr-used
by a postcondition. It must be const on every redeclaration, including those
without written contracts. e.g.,

```C++
int f(const int x) post(r: r >= x);
int f(int x); // Illegal, even though this declaration has no written contracts.

int g(int x) post(sizeof(x) > 0); // OK, no odr-use of x.
int h(int &x) post(x > 0);       // OK, x is a reference parameter.

auto l = [](int x) post(x > 0) { return x; }; // Illegal, x should be const.
```

The implicit const qualification of references in the predicate doesn't make
the parameter declaration const. So checking the type of the `DeclRefExpr`
is not enough. We need the corresponding `ParmVarDecl` and the odr-use
information from expression analysis.

When visiting nested lambdas, only parameters of the function owning this
postcondition should be checked. A lambda parameter with the same name or
parameter index is a different declaration. We also need to use the adjusted
parameter type: `const int p[]` becomes `const int *`, whose pointer is not
const.

[dcl.contract.func, paragraph 7](https://eel.is/c++draft/dcl.contract.func#7)
requires:

> that parameter and the corresponding parameter on all declarations of f
> shall have const type

### Implicit lambda captures

For each implicit capture, Sema should record whether the entity is also
referenced outside the lambda's contracts. We check this after analyzing the
whole lambda body. e.g.,

```C++
void test(int x) {
  auto a = [=]() pre(x > 0) {};             // Illegal.
  auto b = [x]() pre(x > 0) {};             // OK, explicit capture.
  auto c = [=]() pre(x > 0) { return x; };  // OK, also referenced in the body.
  auto d = [=]() { contract_assert(x > 0); }; // Illegal.
}
```

In `a` and `d`, the only reason to capture `x` is the contract assertion. This
is not allowed. In `c`, the body also needs the capture. If we diagnose the
implicit capture as soon as we see `pre(x > 0)`, we will incorrectly reject
`c` before seeing its body.

This check uses potential references, not just odr-uses. It also needs to
account for captures required by nested lambdas. See
[expr.prim.lambda.closure, paragraph 10](https://eel.is/c++draft/expr.prim.lambda.closure#10):

> Adding a contract assertion to an existing C++ program cannot cause
> additional captures.

### Implicit member access

Sema should diagnose implicit access to the current object in a constructor's
precondition or a destructor's postcondition when building the implicit member
access. We shouldn't reject every `CXXThisExpr` found in these predicates.
e.g.,

```C++
struct S {
  int value;
  S() pre(value > 0);                       // Illegal, implicit member access.
  S(int) pre(sizeof(value) > 0);            // OK, unevaluated operand.
  S(double) pre(&this->value != nullptr);   // OK, explicit this and address only.
  ~S() post(value == 0);                    // Illegal, implicit member access.
};
```

The restriction concerns the implicit transformation described in
[expr.prim.id.general, paragraph 2](https://eel.is/c++draft/expr.prim.id.general#2):

> the id-expression is transformed into a class member access expression
> using (*this) as the object expression.

So we need to distinguish this transformation from explicit uses of `this`
and from unevaluated references. For dependent member accesses, the check
runs when the transformation happens during instantiation.

### Comparing redeclarations

The redeclaration can have semantically equal contracts with parameter mappings.
e.g.,

```C++
void f(int x) pre(x > 0);
void f(int y) pre(y > 0); // OK, the parameter was renamed.
```

Here we should think `void f(int x) pre(x > 0);` are semantically equal
with `void f(int y) pre(y > 0)`.

But our current tools to decide declaration/expression similarity
doesn't support such mapping semantics.

We think, in the early versions, we should use our current existing
ODRHash mechanism to decide the contracts equality.

## ABI: Contract Violation Object and Handler

Contract Violation Object and Handler is critical to ABI. This section
we tried to describe the ABI choice of GCC first and then try to figure
out what's we wanted. Note GCC's implementation and ABI of contracts
is still experimental.

Note again that our ABI won't be stable during the developing process
so that the discussing here won't block us developing other parts
of contracts.

### GCC/libstdc++'s implementation choice

The standard specifies the C++ interface, but doesn't specify the ABI between
the compiler and the standard library. The
[current draft](https://eel.is/c++draft/support.contract) specifies
`std::contracts::assertion_kind`, `std::contracts::evaluation_semantic`,
`std::contracts::detection_mode`, `std::contracts::contract_violation`, and
`std::contracts::invoke_default_contract_violation_handler`. It also specifies
the global handler interface:

```C++
void handle_contract_violation(
    const std::contracts::contract_violation &violation);
```

The implementation provides a default definition of this function, but
doesn't declare it in a standard header. **Whether a program can replace it is
implementation-defined.** If replacement is supported, every contract
violation in the program needs to reach the same replacement.

The standard doesn't specify the data members of `contract_violation`, the
underlying types of the enumerations, how the object is constructed, or the
symbol used between generated code and the runtime. It only requires that
the object live throughout the handler invocation. If predicate evaluation
throws, the handler is called while the implicit catch for that exception is
active. The object must not be allocated through a global allocation
function.

[P2811R6, section 3.9](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p2811r6.pdf)
describes two possible implementations. The compiler can construct the C++
object directly, or it can emit versioned data which a library adapter
exposes through the C++ interface. Both are valid implementations.

The current GCC implementation uses the first model. GCC builds an internal
type which has the same layout as libstdc++'s `contract_violation`. The
libstdc++ layout it mirrors can be described as follows:

```C++
namespace std::contracts {
class contract_violation {
  uint16_t Version; // Layout version. Currently 1.

  uint16_t AssertionKind; // std::contracts::assertion_kind:
                          // pre = 1, post = 2, assert = 3.

  uint16_t EvaluationSemantic; // std::contracts::evaluation_semantic:
                               // ignore = 1, observe = 2, enforce = 3,
                               // quick_enforce = 4.

  uint16_t DetectionMode; // std::contracts::detection_mode:
                          // predicate_false = 1, evaluation_exception = 2.

  const char *Comment; // Text describing the violated predicate.

  const void *SourceLocation; // Pointer to libstdc++'s private representation
                              // of std::source_location.

  void *VendorExtension; // Optional implementation-defined data. Currently null.
};
} // namespace std::contracts
```

GCC emits an object with this layout and calls
`::handle_contract_violation` directly. See
[GCC's contracts.cc](https://github.com/gcc-mirror/gcc/blob/master/gcc/cp/contracts.cc)
and [libstdc++'s contracts header](https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/include/std/contracts).

libstdc++ provides a weak default handler and
`invoke_default_contract_violation_handler`. Its out-of-line implementation
is currently provided by `libstdc++exp`. See
[contract26.cc](https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/src/experimental/contract26.cc).
The object layout, source-location representation, enum widths and handler
symbols together form the current GCC/libstdc++ ABI.

For every contract assertion which can call the handler, GCC normally emits
a TU-local read-only violation object. This avoids dynamic initialization and
runtime construction, and makes the violation path simple. The cost is that a
program with many contract assertions or template instantiations can contain
many complete violation objects, source-location records and relocations. It
also makes the compiler depend on libstdc++'s private object and
source-location layouts.

For example, given an enforced precondition which may throw:

```C++
bool valid(int);
void f(int x) pre(valid(x)) {
  // function body
}
```

GCC conceptually generates code like this:

```C++
static const LibstdcxxSourceLocationImpl FPreLocation = {
  "file.cpp", "f", 2, 15
};

static const std::contracts::contract_violation FPreViolation =
    {
        /* Version */            1,
        /* AssertionKind */      std::contracts::assertion_kind::pre,
        /* EvaluationSemantic */ std::contracts::evaluation_semantic::enforce,
        /* DetectionMode */      std::contracts::detection_mode::predicate_false,
        /* Comment */            "valid(x)",
        /* SourceLocation */     &FPreLocation,
        /* VendorExtension */    nullptr
    };

void f(int x) {
  bool Failed;
  try {
    Failed = !valid(x);
  } catch (...) {
    std::contracts::contract_violation ExceptionViolation =
      FPreViolation; // compiler-internal copy
    ExceptionViolation.DetectionMode =
      std::contracts::detection_mode::evaluation_exception;
    ::handle_contract_violation(ExceptionViolation);
    std::terminate();
  }

  if (Failed) {
    ::handle_contract_violation(FPreViolation);
    std::terminate();
  }

  // function body
}
```

### What we are going to do

As we won't claim ABI's stability, we propose we should follow GCC's current
ABI in the first version. This is simple and easier to implement. What's more
important is, it keeps the ABI compatibility with GCC. This is very important
for the whole C++ community.

However, this is not saying we should follow everything GCC did. We should
discuss the ABI of contracts in an ABI standard group and finalize the ABI
there. We can do that parallelly with our initial implementation of contracts.

## CodeGen

The function entry and exit paths decide when to evaluate `pre` and `post`, 
while `contract_assert` is emitted at its position in the body. They should 
share the same predicate emission.

CodeGen consumes the contracts stored on the current `FunctionDecl`. As mentioned
above, CodeGen doesn't have to search the redeclaration chain or instantiate
predicates.

### Emitting a violation

For each contract which can call the handler, CodeGen emits a TU-local
read-only internal record with the GCC layout described above. The record
contains the selected evaluation semantic, predicate text and a pointer to a
layout-compatible source-location record. It is emitted only for `observe` and
`enforce`; `ignore` and `quick_enforce` don't reference the handler.

The predicate-false path passes this object directly to
`::handle_contract_violation`. The evaluation-exception path makes a stack
copy, changes its detection mode to `evaluation_exception`, and passes the
copy. We must not modify the read-only object because the same assertion can
fail concurrently on several threads.

The call should use Clang's normal C++ mangling and calling-convention
machinery. The generated code must call the replaceable global handler
rather than `invoke_default_contract_violation_handler`; the library's
weak default definition provides the fallback.

### Where to emit checks

We should emit a function's own preconditions and postconditions in the
callee. This keeps the checks available through function pointers and when
the caller doesn't see the contracts. e.g.,

```C++
extern "C" int positive(int x) pre(x > 0) {
  return x;
}
```

```C
int positive(int);
int use_positive(void) {
  return positive(1); // The C++ callee performs the check.
}
```

The function keeps its ordinary symbol, calling convention and parameter and
return ABI. We don't need a checked and unchecked public entry point, or
hidden arguments carrying contract information. A definition compiled with
`ignore` won't gain checks just because its caller enables them.

### Virtual calls

Always emitting contracts in the callee can't handle the case for virtual
calls.

But for simplicity, we can skip the case of virtual calls in the first
version.

### Postconditions and cleanups

We should emit preconditions after the function parameters are available,
before starting the body. For postconditions, we need a normal-exit cleanup
on `EHScopeStack`, placed above parameter cleanups and below body-local
cleanups. It runs after the return value is initialized and the body's local
variables are destroyed, but before the parameters are destroyed. e.g.,

```C++
struct SetOnExit {
  int &value;
  ~SetOnExit() { value = 1; }
};

void f(int &value) post(value == 1) {
  SetOnExit guard{value};
  return;
}
```

Here the postcondition must run after `guard` is destroyed. Emitting the
check directly at the return statement would be too early. Emitting it in
the final epilogue after all cleanups would be too late for postconditions
which refer to parameters with destructors.

This follows the ordering in
[stmt.return, paragraph 5](https://eel.is/c++draft/stmt.return#5), and
[expr.call, paragraph 10](https://eel.is/c++draft/expr.call#10) requires:

> These evaluations, in turn, are sequenced before the destruction of any
> function parameters.

The cleanup is normal-only: it handles early returns and falling off the end
of a void function, but doesn't evaluate postconditions while unwinding out
of the body. Within each entry or exit phase, we visit the corresponding
contracts in source order.

A `ResultNameDecl` doesn't introduce storage. While emitting a postcondition,
we should map it to the source-level object in the existing return storage,
before the result is converted to its ABI representation. For a reference
return, this means the referred-to object, not the return slot which contains
its address.

For a constructor, the generated base and member initializers are part of the
function body for contract ordering. For a destructor, the generated base and
member destructors are part of the body. We should therefore emit the four
phases as follows:

```text
constructor preconditions
base and member initialization
constructor compound-statement
constructor postconditions

destructor preconditions
destructor compound-statement
member and base destruction
destructor postconditions
```

Emitting checks merely around the user-written compound-statement would put a
constructor's preconditions after initialization and a destructor's
postconditions before destruction.

In the Itanium ABI, one source-level constructor or destructor can have several
entry points. For example:

```C++
struct V {};

bool cond();

struct S : virtual V {
  S(int value) pre(cond()) post(cond());
  ~S() pre(cond()) post(cond());
};
```

Conceptually, the generated code in the Itanium ABI looks like:

```text
S::S C1      = check preconditions
                + initialize virtual bases
                + perform the C2 portion without its checks
                + check postconditions
S::S C2      = check preconditions
                + initialize non-virtual bases and members
                + run the body
                + check postconditions

S::~S D0     = call D1 + operator delete
S::~S D1     = check preconditions
                + perform the D2 portion without its checks
                + destroy virtual bases
                + check postconditions
S::~S D2     = check preconditions
                + run the body
                + destroy members and non-virtual bases
                + check postconditions
```

Clang may forward `C1` to `C2` and forward `D1` to `D2` as an optimization
to decrease the code size. However, if contracts are involved, we can't do
such forwarding because it may cause the contract checks to be evaluated
multiple times.

### Predicate evaluation and exceptions

#### Pairing contracts handler and exception handlers

When the contracts handler throws, we should be able to get the
correct exception being throwing.

So we should generate code look like:

```C++
bool Failed;

try {
  Failed = !predicate();
} catch (...) {
  ::handle_contract_violation(EvaluationExceptionViolation);
  if (Semantic == enforce)
    std::terminate();
  goto Done; // observe
}

if (Failed) {
  ::handle_contract_violation(PredicateFalseViolation);
  if (Semantic == enforce)
    std::terminate();
}

Done:;
```

instead of

```C++
try {
    if (!predicate()) {
      ::handle_contract_violation(PredicateFalseViolation);

      if (Semantic == enforce)
        contract_terminate();
    }
  } catch (...) {
    ::handle_contract_violation(EvaluationExceptionViolation);

    if (Semantic == enforce)
      contract_terminate();
  }
```

As if `::handle_contract_violation` throws, the exception from `::handle_contract_violation`
may be treated as an exception from the `predicate`.

#### Lifetime of the result variable if postcondition throws

Another problem is that a postcondition's handler can throw after the
result has already been constructed. For example, assume that the violation
handler throws:

```C++
int Alive = 0;

struct Result {
  Result() { ++Alive; }
  ~Result() { --Alive; }

  ...
};

Result make() post(false) {
  //... logics in make
  return Result{};
}

void use() {
  try {
    Result result = make();
  } catch (...) {
    // Alive must be 0 here.
  }
}
```

The result has been constructed when the postcondition runs, but the
initialization of `result` doesn't complete when the handler throws. The
caller therefore won't run its normal destructor. So that if we generate
code like:

```C++
// Assuming we pass the result in the return address slot
void make(Result *__result) {
  // ... logics in make

  // Construct Result at the specified address
  new (__result) Result;

  if (!postcondition_predicate()) {
    // If the handler throws
    ::handle_contract_violation(PostconditionViolation);
  }

  return;
}
```

then the caller of `make` may not destruct `Result` as the initialization
of `Result` is not completed. Then `Result` may be leaked.

So that we need to handle the lifetime of the result in `make` correctly:

```C++
void make(Result *__result) {
  //... logics in make

  bool ResultConstructed = false;

  try {
    new (__result) Result;
    ResultConstructed = true;

    if (!postcondition_predicate()) {
      ::handle_contract_violation(PostconditionViolation);
    }

    ResultConstructed = false;
    return;
  } catch (...) {
    if (ResultConstructed)
      __result->~Result();

    throw;
  }
}
```

The `try`/`catch` here are pesudo code. We should use `EHScopeStack` in
CodeGen.
