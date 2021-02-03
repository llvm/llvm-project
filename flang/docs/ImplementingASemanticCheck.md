<!--===- docs/ImplementingASemanticCheck.md 
  
   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  
-->
# How to implement a Sematic Check in Flang

```eval_rst
.. contents::
   :local:
```

I recently added a semantic check to the f18 compiler front end.  This document
describes my thought process and the resulting implementation.

For more information about the compiler, start with the 
[compiler overview](Overview.md).

## Problem definition

In the 2018 Fortran standard, section 11.1.7.4.3, paragraph 2, states that:

```
Except for the incrementation of the DO variable that occurs in step (3), the DO variable 
shall neither be redefined nor become undefined while the DO construct is active.
```
One of the ways that DO variables might be redefined is if they are passed to
functions with dummy arguments whose `INTENT` is `INTENT(OUT)` or
`INTENT(INOUT)`.  I implemented this semantic check.  Specifically, I changed
the compiler to emit an error message if an active DO variable was passed to a
dummy argument of a FUNCTION with INTENT(OUT).  Similarly, I had the compiler
emit a warning if an active DO variable was passed to a dummy argument with
INTENT(INOUT).  Previously, I had implemented similar checks for SUBROUTINE
calls.

## Creating a test

My first step was to create a test case to cause the problem.  I called it testfun.f90 and used it to check the behavior of other Fortran compilers.  Here's the initial version:

```fortran
  subroutine s()
    Integer :: ivar, jvar

    do ivar = 1, 10
      jvar = intentOutFunc(ivar) ! Error since ivar is a DO variable
    end do

  contains
    function intentOutFunc(dummyArg)
      integer, intent(out) :: dummyArg
      integer  :: intentOutFunc

      dummyArg = 216
    end function intentOutFunc
  end subroutine s
```

I verified that other Fortran compilers produced an error message at the point
of the call to `intentOutFunc()`:

```fortran
      jvar = intentOutFunc(ivar) ! Error since ivar is a DO variable
```


I also used this program to produce a parse tree for the program using the command:
```bash
  f18 -fdebug-dump-parse-tree -fsyntax-only testfun.f90
```

Here's the relevant fragment of the parse tree produced by the compiler:

```
| | ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct
| | | NonLabelDoStmt
| | | | LoopControl -> LoopBounds
| | | | | Scalar -> Name = 'ivar'
| | | | | Scalar -> Expr = '1_4'
| | | | | | LiteralConstant -> IntLiteralConstant = '1'
| | | | | Scalar -> Expr = '10_4'
| | | | | | LiteralConstant -> IntLiteralConstant = '10'
| | | Block
| | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'jvar=intentoutfunc(ivar)'
| | | | | Variable -> Designator -> DataRef -> Name = 'jvar'
| | | | | Expr = 'intentoutfunc(ivar)'
| | | | | | FunctionReference -> Call
| | | | | | | ProcedureDesignator -> Name = 'intentoutfunc'
| | | | | | | ActualArgSpec
| | | | | | | | ActualArg -> Expr = 'ivar'
| | | | | | | | | Designator -> DataRef -> Name = 'ivar'
| | | EndDoStmt -> 
```

Note that this fragment of the tree only shows four `parser::Expr` nodes,
but the full parse tree also contained a fifth `parser::Expr` node for the
constant 216 in the statement:

```fortran
      dummyArg = 216
```
## Analysis and implementation planning

I then considered what I needed to do.  I needed to detect situations where an
active DO variable was passed to a dummy argument with `INTENT(OUT)` or
`INTENT(INOUT)`.  Once I detected such a situation, I needed to produce a
message that highlighted the erroneous source code.  

### Deciding where to add the code to the compiler
This new semantic check would depend on several types of information -- the
parse tree, source code location information, symbols, and expressions.  Thus I
needed to put my new code in a place in the compiler after the parse tree had
been created, name resolution had already happened, and expression semantic
checking had already taken place.

Most semantic checks for statements are implemented by walking the parse tree
and performing analysis on the nodes they visit.  My plan was to use this
method.  The infrastructure for walking the parse tree for statement semantic
checking is implemented in the files `lib/Semantics/semantics.cpp`.
Here's a fragment of the declaration of the framework's parse tree visitor from
`lib/Semantics/semantics.cpp`:

```C++
  // A parse tree visitor that calls Enter/Leave functions from each checker
  // class C supplied as template parameters. Enter is called before the node's
  // children are visited, Leave is called after. No two checkers may have the
  // same Enter or Leave function. Each checker must be constructible from
  // SemanticsContext and have BaseChecker as a virtual base class.
  template<typename... C> class SemanticsVisitor : public virtual C... {
  public:
    using C::Enter...;
    using C::Leave...;
    using BaseChecker::Enter;
    using BaseChecker::Leave;
    SemanticsVisitor(SemanticsContext &context)
      : C{context}..., context_{context} {}
      ...

```

Since FUNCTION calls are a kind of expression, I was planning to base my
implementation on the contents of `parser::Expr` nodes.  I would need to define
either an `Enter()` or `Leave()` function whose parameter was a `parser::Expr`
node.  Here's the declaration I put into `lib/Semantics/check-do.h`:

```C++
  void Leave(const parser::Expr &);
```
The `Enter()` functions get called at the time the node is first visited --
that is, before its children.  The `Leave()` function gets called after the
children are visited.  For my check the visitation order didn't matter, so I
arbitrarily chose to implement the `Leave()` function to visit the parse tree
node.

Since my semantic check was focused on DO CONCURRENT statements, I added it to
the file `lib/Semantics/check-do.cpp` where most of the semantic checking for
DO statements already lived.

### Taking advantage of prior work
When implementing a similar check for SUBROUTINE calls, I created a utility
functions in `lib/Semantics/semantics.cpp` to emit messages if
a symbol corresponding to an active DO variable was being potentially modified:

```C++
  void WarnDoVarRedefine(const parser::CharBlock &location, const Symbol &var);
  void CheckDoVarRedefine(const parser::CharBlock &location, const Symbol &var);
```

The first function is intended for dummy arguments of `INTENT(INOUT)` and
the second for `INTENT(OUT)`.

Thus I needed three pieces of
information -- 
1. the source location of the erroneous text, 
2. the `INTENT` of the associated dummy argument, and
3. the relevant symbol passed as the actual argument.

The first and third are needed since they're required to call the utility
functions.  The second is needed to determine whether to call them.

### Finding the source location
The source code location information that I'd need for the error message must
come from the parse tree.  I looked in the file
`include/flang/Parser/parse-tree.h` and determined that a `struct Expr`
contained source location information since it had the field `CharBlock
source`.  Thus, if I visited a `parser::Expr` node, I could get the source
location information for the associated expression.

### Determining the `INTENT`
I knew that I could find the `INTENT` of the dummy argument associated with the
actual argument from the function called `dummyIntent()` in the class
`evaluate::ActualArgument` in the file `include/flang/Evaluate/call.h`.  So
if I could find an `evaluate::ActualArgument` in an expression, I could
  determine the `INTENT` of the associated dummy argument.  I knew that it was
  valid to call `dummyIntent()` because the data on which `dummyIntent()`
  depends is established during semantic processing for expressions, and the
  semantic processing for expressions happens before semantic checking for DO
  constructs.

In my prior work on checking the INTENT of arguments for SUBROUTINE calls,
the parse tree held a node for the call (a `parser::CallStmt`) that contained
an `evaluate::ProcedureRef` node.
```C++
  struct CallStmt {
    WRAPPER_CLASS_BOILERPLATE(CallStmt, Call);
    mutable std::unique_ptr<evaluate::ProcedureRef,
        common::Deleter<evaluate::ProcedureRef>>
        typedCall;  // filled by semantics
  };
```
The `evaluate::ProcedureRef` contains a list of `evaluate::ActualArgument`
nodes.  I could then find the INTENT of a dummy argument from the
`evaluate::ActualArgument` node.

For a FUNCTION call, though, there is no similar way to get from a parse tree
node to an `evaluate::ProcedureRef` node.  But I knew that there was an
existing framework used in DO construct semantic checking that traversed an
`evaluate::Expr` node collecting `semantics::Symbol` nodes.  I guessed that I'd
be able to use a similar framework to traverse an `evaluate::Expr`  node to
find all of the `evaluate::ActualArgument` nodes.  

Note that the compiler has multiple types called `Expr`.  One is in the
`parser` namespace.  `parser::Expr` is defined in the file
`include/flang/Parser/parse-tree.h`.  It represents a parsed expression that
maps directly to the source code and has fields that specify any operators in
the expression, the operands, and the source position of the expression.

Additionally, in the namespace `evaluate`, there are `evaluate::Expr<T>`
template classes defined in the file `include/flang/Evaluate/expression.h`.
These are parameterized over the various types of Fortran and constitute a
suite of strongly-typed representations of valid Fortran expressions of type
`T` that have been fully elaborated with conversion operations and subjected to
constant folding.  After an expression has undergone semantic analysis, the
field `typedExpr` in the `parser::Expr` node is filled in with a pointer that
owns an instance of `evaluate::Expr<SomeType>`, the most general representation
of an analyzed expression.

All of the declarations associated with both FUNCTION and SUBROUTINE calls are
in `include/flang/Evaluate/call.h`.  An `evaluate::FunctionRef` inherits from
an `evaluate::ProcedureRef` which contains the list of
`evaluate::ActualArgument` nodes.  But the relationship between an
`evaluate::FunctionRef` node and its associated arguments is not relevant.  I
only needed to find the `evaluate::ActualArgument` nodes in an expression.
They hold all of the information I needed.

So my plan was to start with the `parser::Expr` node and extract its
associated `evaluate::Expr` field.  I would then traverse the
`evaluate::Expr` tree collecting all of the `evaluate::ActualArgument`
nodes.  I would look at each of these nodes to determine the `INTENT` of
the associated dummy argument.

This combination of the traversal framework and `dummyIntent()` would give
me the `INTENT` of all of the dummy arguments in a FUNCTION call.  Thus, I
would have the second piece of information I needed.

### Determining if the actual argument is a variable
I also guessed that I could determine if the `evaluate::ActualArgument`
consisted of a variable.  

Once I had a symbol for the variable, I could call one of the functions:
```C++
  void WarnDoVarRedefine(const parser::CharBlock &, const Symbol &);
  void CheckDoVarRedefine(const parser::CharBlock &, const Symbol &);
```
to emit the messages.

If my plans worked out, this would give me the three pieces of information I
needed -- the source location of the erroneous text, the `INTENT` of the dummy
argument, and a symbol that I could use to determine whether the actual
argument was an active DO variable.

## Implementation

### Adding a parse tree visitor
I started my implementation by adding a visitor for `parser::Expr` nodes.
Since this analysis is part of DO construct checking, I did this in
`lib/Semantics/check-do.cpp`.  I added a print statement to the visitor to
verify that my new code was actually getting executed.  

In `lib/Semantics/check-do.h`, I added the declaration for the visitor:

```C++
  void Leave(const parser::Expr &);
```

In `lib/Semantics/check-do.cpp`, I added an (almost empty) implementation:

```C++
  void DoChecker::Leave(const parser::Expr &) {
    std::cout << "In Leave for parser::Expr\n";
  }
```

I then built the compiler with these changes and ran it on my test program.
This time, I made sure to invoke semantic checking.  Here's the command I used:
```bash
  f18 -fdebug-resolve-names -fdebug-dump-parse-tree -funparse-with-symbols testfun.f90
```

This produced the output:

```
  In Leave for parser::Expr
  In Leave for parser::Expr
  In Leave for parser::Expr
  In Leave for parser::Expr
  In Leave for parser::Expr
```

This made sense since the parse tree contained five `parser::Expr` nodes.
So far, so good.  Note that a `parse::Expr` node has a field with the
source position of the associated expression (`CharBlock source`).  So I
now had one of the three pieces of information needed to detect and report
errors.

### Collecting the actual arguments
To get the `INTENT` of the dummy arguments and the `semantics::Symbol` associated with the
actual argument, I needed to find all of the actual arguments embedded in an
expression that contained a FUNCTION call.  So my next step was to write the
framework to walk the `evaluate::Expr` to gather all of the
`evaluate::ActualArgument` nodes.  The code that I planned to model it on
was the existing infrastructure that collected all of the `semantics::Symbol` nodes from an
`evaluate::Expr`.  I found this implementation in
`lib/Evaluate/tools.cpp`:

```C++
  struct CollectSymbolsHelper
    : public SetTraverse<CollectSymbolsHelper, semantics::SymbolSet> {
    using Base = SetTraverse<CollectSymbolsHelper, semantics::SymbolSet>;
    CollectSymbolsHelper() : Base{*this} {}
    using Base::operator();
    semantics::SymbolSet operator()(const Symbol &symbol) const {
      return {symbol};
    }
  };
  template<typename A> semantics::SymbolSet CollectSymbols(const A &x) {
    return CollectSymbolsHelper{}(x);
  }
```

Note that the `CollectSymbols()` function returns a `semantics::Symbolset`,
which is declared in `include/flang/Semantics/symbol.h`:

```C++
  using SymbolSet = std::set<SymbolRef>;
```

This infrastructure yields a collection based on `std::set<>`.  Using an
`std::set<>` means that if the same object is inserted twice, the
collection only gets one copy.  This was the behavior that I wanted.

Here's a sample invocation of `CollectSymbols()` that I found:
```C++
    if (const auto *expr{GetExpr(parsedExpr)}) {
      for (const Symbol &symbol : evaluate::CollectSymbols(*expr)) {
```

I noted that a `SymbolSet` did not actually contain an
`std::set<Symbol>`.  This wasn't surprising since we don't want to put the
full `semantics::Symbol` objects into the set.  Ideally, we would be able to create an
`std::set<Symbol &>` (a set of C++ references to symbols).  But C++ doesn't
support sets that contain references.  This limitation is part of the rationale
for the f18 implementation of type `common::Reference`, which is defined in
  `include/flang/Common/reference.h`.

`SymbolRef`, the specialization of the template `common::Reference` for
`semantics::Symbol`, is declared in the file
`include/flang/Semantics/symbol.h`:

```C++
  using SymbolRef = common::Reference<const Symbol>;
```

So to implement something that would collect `evaluate::ActualArgument`
nodes from an `evaluate::Expr`, I first defined the required types
`ActualArgumentRef` and `ActualArgumentSet`.  Since these are being
used exclusively for DO construct semantic checking (currently), I put their
definitions into `lib/Semantics/check-do.cpp`:


```C++
  namespace Fortran::evaluate {
    using ActualArgumentRef = common::Reference<const ActualArgument>;
  }


  using ActualArgumentSet = std::set<evaluate::ActualArgumentRef>;
```

Since `ActualArgument` is in the namespace `evaluate`, I put the
definition for `ActualArgumentRef` in that namespace, too.

I then modeled the code to create an `ActualArgumentSet` after the code to
collect a `SymbolSet` and put it into `lib/Semantics/check-do.cpp`:


```C++
  struct CollectActualArgumentsHelper
    : public evaluate::SetTraverse<CollectActualArgumentsHelper,
          ActualArgumentSet> {
    using Base = SetTraverse<CollectActualArgumentsHelper, ActualArgumentSet>;
    CollectActualArgumentsHelper() : Base{*this} {}
    using Base::operator();
    ActualArgumentSet operator()(const evaluate::ActualArgument &arg) const {
      return ActualArgumentSet{arg};
    }
  };

  template<typename A> ActualArgumentSet CollectActualArguments(const A &x) {
    return CollectActualArgumentsHelper{}(x);
  }

  template ActualArgumentSet CollectActualArguments(const SomeExpr &);
```

Unfortunately, when I tried to build this code, I got an error message saying
`std::set` requires the `<` operator to be defined for its contents.
To fix this, I added a definition for `<`.  I didn't care how `<` was
defined, so I just used the address of the object:

```C++
  inline bool operator<(ActualArgumentRef x, ActualArgumentRef y) {
    return &*x < &*y;
  }
```

I was surprised when this did not make the error message saying that I needed
the `<` operator go away.  Eventually, I figured out that the definition of
the `<` operator needed to be in the `evaluate` namespace.  Once I put
it there, everything compiled successfully.  Here's the code that worked:

```C++
  namespace Fortran::evaluate {
  using ActualArgumentRef = common::Reference<const ActualArgument>;

  inline bool operator<(ActualArgumentRef x, ActualArgumentRef y) {
    return &*x < &*y;
  }
  }
```

I then modified my visitor for the parser::Expr to invoke my new collection
framework.  To verify that it was actually doing something, I printed out the
number of `evaluate::ActualArgument` nodes that it collected.  Note the
call to `GetExpr()` in the invocation of `CollectActualArguments()`.  I
modeled this on similar code that collected a `SymbolSet` described above:

```C++
  void DoChecker::Leave(const parser::Expr &parsedExpr) {
    std::cout << "In Leave for parser::Expr\n";
    ActualArgumentSet argSet{CollectActualArguments(GetExpr(parsedExpr))};
    std::cout << "Number of arguments: " << argSet.size() << "\n";
  }
```

I compiled and tested this code on my little test program.  Here's the output that I got:
```
  In Leave for parser::Expr
  Number of arguments: 0
  In Leave for parser::Expr
  Number of arguments: 0
  In Leave for parser::Expr
  Number of arguments: 0
  In Leave for parser::Expr
  Number of arguments: 1
  In Leave for parser::Expr
  Number of arguments: 0
```

So most of the `parser::Expr`nodes contained no actual arguments, but the
fourth expression in the parse tree walk contained a single argument.  This may
seem wrong since the third `parser::Expr` node in the file contains the
`FunctionReference` node along with the arguments that we're gathering.
But since the tree walk function is being called upon leaving a
`parser::Expr` node, the function visits the `parser::Expr` node
associated with the `parser::ActualArg` node before it visits the
`parser::Expr` node associated with the `parser::FunctionReference`
node.

So far, so good.

### Finding the `INTENT` of the dummy argument
I now wanted to find the `INTENT` of the dummy argument associated with the
arguments in the set.  As mentioned earlier, the type
`evaluate::ActualArgument` has a member function called `dummyIntent()`
that gives this value.  So I augmented my code to print out the `INTENT`:

```C++
  void DoChecker::Leave(const parser::Expr &parsedExpr) {
    std::cout << "In Leave for parser::Expr\n";
    ActualArgumentSet argSet{CollectActualArguments(GetExpr(parsedExpr))};
    std::cout << "Number of arguments: " << argSet.size() << "\n";
    for (const evaluate::ActualArgumentRef &argRef : argSet) {
      common::Intent intent{argRef->dummyIntent()};
      switch (intent) {
        case common::Intent::In: std::cout << "INTENT(IN)\n"; break;
        case common::Intent::Out: std::cout << "INTENT(OUT)\n"; break;
        case common::Intent::InOut: std::cout << "INTENT(INOUT)\n"; break;
        default: std::cout << "default INTENT\n";
      }
    }
  }
```

I then rebuilt my compiler and ran it on my test case.  This produced the following output:

```
  In Leave for parser::Expr
  Number of arguments: 0
  In Leave for parser::Expr
  Number of arguments: 0
  In Leave for parser::Expr
  Number of arguments: 0
  In Leave for parser::Expr
  Number of arguments: 1
  INTENT(OUT)
  In Leave for parser::Expr
  Number of arguments: 0
```

I then modified my test case to convince myself that I was getting the correct
`INTENT` for `IN`, `INOUT`, and default cases.

So far, so good.

### Finding the symbols for arguments that are variables
The third and last piece of information I needed was to determine if a variable
was being passed as an actual argument.  In such cases, I wanted to get the
symbol table node (`semantics::Symbol`) for the variable.  My starting point was the
`evaluate::ActualArgument` node.  

I was unsure of how to do this, so I browsed through existing code to look for
how it treated `evaluate::ActualArgument` objects.  Since most of the code that deals with the `evaluate` namespace is in the lib/Evaluate directory, I looked there.  I ran `grep` on all of the `.cpp` files looking for
uses of `ActualArgument`.  One of the first hits I got was in `lib/Evaluate/call.cpp` in the definition of `ActualArgument::GetType()`:

```C++
std::optional<DynamicType> ActualArgument::GetType() const {
  if (const Expr<SomeType> *expr{UnwrapExpr()}) {
    return expr->GetType();
  } else if (std::holds_alternative<AssumedType>(u_)) {
    return DynamicType::AssumedType();
  } else {
    return std::nullopt;
  }
}
```

I noted the call to `UnwrapExpr()` that yielded a value of
`Expr<SomeType>`.  So I guessed that I could use this member function to
get an `evaluate::Expr<SomeType>` on which I could perform further analysis.

I also knew that the header file `include/flang/Evaluate/tools.h` held many
utility functions for dealing with `evaluate::Expr` objects.  I was hoping to
find something that would determine if an `evaluate::Expr` was a variable.  So
I searched for `IsVariable` and got a hit immediately.  
```C++
  template<typename A> bool IsVariable(const A &x) {
    if (auto known{IsVariableHelper{}(x)}) {
      return *known;
    } else {
      return false;
    }
  }
```

But I actually needed more than just the knowledge that an `evaluate::Expr` was
a variable.  I needed the `semantics::Symbol` associated with the variable.  So
I searched in `include/flang/Evaluate/tools.h` for functions that returned a
`semantics::Symbol`.  I found the following:

```C++
// If an expression is simply a whole symbol data designator,
// extract and return that symbol, else null.
template<typename A> const Symbol *UnwrapWholeSymbolDataRef(const A &x) {
  if (auto dataRef{ExtractDataRef(x)}) {
    if (const SymbolRef * p{std::get_if<SymbolRef>(&dataRef->u)}) {
      return &p->get();
    }
  }
  return nullptr;
}
```

This was exactly what I wanted.  DO variables must be whole symbols.  So I
could try to extract a whole `semantics::Symbol` from the `evaluate::Expr` in my
`evaluate::ActualArgument`.  If this extraction resulted in a `semantics::Symbol`
that wasn't a `nullptr`, I could then conclude if it was a variable that I
could pass to existing functions that would determine if it was an active DO
variable.

I then modified the compiler to perform the analysis that I'd guessed would
work:

```C++
  void DoChecker::Leave(const parser::Expr &parsedExpr) {
    std::cout << "In Leave for parser::Expr\n";
    ActualArgumentSet argSet{CollectActualArguments(GetExpr(parsedExpr))};
    std::cout << "Number of arguments: " << argSet.size() << "\n";
    for (const evaluate::ActualArgumentRef &argRef : argSet) {
      if (const SomeExpr * argExpr{argRef->UnwrapExpr()}) {
        std::cout << "Got an unwrapped Expr\n";
        if (const Symbol * var{evaluate::UnwrapWholeSymbolDataRef(*argExpr)}) {
          std::cout << "Found a whole variable: " << *var << "\n";
        }
      }
      common::Intent intent{argRef->dummyIntent()};
      switch (intent) {
        case common::Intent::In: std::cout << "INTENT(IN)\n"; break;
        case common::Intent::Out: std::cout << "INTENT(OUT)\n"; break;
        case common::Intent::InOut: std::cout << "INTENT(INOUT)\n"; break;
        default: std::cout << "default INTENT\n";
      }
    }
  }
```  

Note the line that prints out the symbol table entry for the variable:

```C++
          std::cout << "Found a whole variable: " << *var << "\n";
```  

The compiler defines the "<<" operator for `semantics::Symbol`, which is handy
for analyzing the compiler's behavior.

Here's the result of running the modified compiler on my Fortran test case:

```
  In Leave for parser::Expr
  Number of arguments: 0
  In Leave for parser::Expr
  Number of arguments: 0
  In Leave for parser::Expr
  Number of arguments: 0
  In Leave for parser::Expr
  Number of arguments: 1
  Got an unwrapped Expr
  Found a whole variable: ivar: ObjectEntity type: INTEGER(4)
  INTENT(OUT)
  In Leave for parser::Expr
  Number of arguments: 0
```

Sweet.

### Emitting the messages
At this point, using the source location information from the original
`parser::Expr`, I had enough information to plug into the exiting
interfaces for emitting messages for active DO variables.  I modified the
compiler code accordingly:


```C++
  void DoChecker::Leave(const parser::Expr &parsedExpr) {
    std::cout << "In Leave for parser::Expr\n";
    ActualArgumentSet argSet{CollectActualArguments(GetExpr(parsedExpr))};
    std::cout << "Number of arguments: " << argSet.size() << "\n";
    for (const evaluate::ActualArgumentRef &argRef : argSet) {
      if (const SomeExpr * argExpr{argRef->UnwrapExpr()}) {
        std::cout << "Got an unwrapped Expr\n";
        if (const Symbol * var{evaluate::UnwrapWholeSymbolDataRef(*argExpr)}) {
          std::cout << "Found a whole variable: " << *var << "\n";
          common::Intent intent{argRef->dummyIntent()};
          switch (intent) {
            case common::Intent::In: std::cout << "INTENT(IN)\n"; break;
            case common::Intent::Out: 
              std::cout << "INTENT(OUT)\n"; 
              context_.CheckDoVarRedefine(parsedExpr.source, *var);
              break;
            case common::Intent::InOut: 
              std::cout << "INTENT(INOUT)\n"; 
              context_.WarnDoVarRedefine(parsedExpr.source, *var);
              break;
            default: std::cout << "default INTENT\n";
          }
        }
      }
    }
  }
```  

I then ran this code on my test case, and miraculously, got the following
output:

```
  In Leave for parser::Expr
  Number of arguments: 0
  In Leave for parser::Expr
  Number of arguments: 0
  In Leave for parser::Expr
  Number of arguments: 0
  In Leave for parser::Expr
  Number of arguments: 1
  Got an unwrapped Expr
  Found a whole variable: ivar: ObjectEntity type: INTEGER(4)
  INTENT(OUT)
  In Leave for parser::Expr
  Number of arguments: 0
  testfun.f90:6:12: error: Cannot redefine DO variable 'ivar'
        jvar = intentOutFunc(ivar)
               ^^^^^^^^^^^^^^^^^^^
  testfun.f90:5:6: Enclosing DO construct
      do ivar = 1, 10
         ^^^^
```

Even sweeter.

## Improving the test case
At this point, my implementation seemed to be working.  But I was concerned
about the limitations of my test case.  So I augmented it to include arguments
other than `INTENT(OUT)` and more complex expressions.  Luckily, my
augmented test did not reveal any new problems.   

Here's the test I ended up with:

```Fortran
  subroutine s()

    Integer :: ivar, jvar

    ! This one is OK
    do ivar = 1, 10
      jvar = intentInFunc(ivar)
    end do

    ! Error for passing a DO variable to an INTENT(OUT) dummy
    do ivar = 1, 10
      jvar = intentOutFunc(ivar)
    end do

    ! Error for passing a DO variable to an INTENT(OUT) dummy, more complex 
    ! expression
    do ivar = 1, 10
      jvar = 83 + intentInFunc(intentOutFunc(ivar))
    end do

    ! Warning for passing a DO variable to an INTENT(INOUT) dummy
    do ivar = 1, 10
      jvar = intentInOutFunc(ivar)
    end do

  contains
    function intentInFunc(dummyArg)
      integer, intent(in) :: dummyArg
      integer  :: intentInFunc

      intentInFunc = 343
    end function intentInFunc

    function intentOutFunc(dummyArg)
      integer, intent(out) :: dummyArg
      integer  :: intentOutFunc

      dummyArg = 216
      intentOutFunc = 343
    end function intentOutFunc

    function intentInOutFunc(dummyArg)
      integer, intent(inout) :: dummyArg
      integer  :: intentInOutFunc

      dummyArg = 216
      intentInOutFunc = 343
    end function intentInOutFunc

  end subroutine s
```

## Submitting the pull request
At this point, my implementation seemed functionally complete, so I stripped out all of the debug statements, ran `clang-format` on it and reviewed it
to make sure that the names were clear.  Here's what I ended up with:

```C++
  void DoChecker::Leave(const parser::Expr &parsedExpr) {
    ActualArgumentSet argSet{CollectActualArguments(GetExpr(parsedExpr))};
    for (const evaluate::ActualArgumentRef &argRef : argSet) {
      if (const SomeExpr * argExpr{argRef->UnwrapExpr()}) {
        if (const Symbol * var{evaluate::UnwrapWholeSymbolDataRef(*argExpr)}) {
          common::Intent intent{argRef->dummyIntent()};
          switch (intent) {
            case common::Intent::Out: 
              context_.CheckDoVarRedefine(parsedExpr.source, *var);
              break;
            case common::Intent::InOut: 
              context_.WarnDoVarRedefine(parsedExpr.source, *var);
              break;
            default:; // INTENT(IN) or default intent
          }
        }
      }
    }
  }
```

I then created a pull request to get review comments.  

## Responding to pull request comments
I got feedback suggesting that I use an `if` statement rather than a
`case` statement.  Another comment reminded me that I should look at the
code I'd previously writted to do a similar check for SUBROUTINE calls to see
if there was an opportunity to share code.  This examination resulted in
  converting my existing code to the following pair of functions:


```C++
  static void CheckIfArgIsDoVar(const evaluate::ActualArgument &arg,
      const parser::CharBlock location, SemanticsContext &context) {
    common::Intent intent{arg.dummyIntent()};
    if (intent == common::Intent::Out || intent == common::Intent::InOut) {
      if (const SomeExpr * argExpr{arg.UnwrapExpr()}) {
        if (const Symbol * var{evaluate::UnwrapWholeSymbolDataRef(*argExpr)}) {
          if (intent == common::Intent::Out) {
            context.CheckDoVarRedefine(location, *var);
          } else {
            context.WarnDoVarRedefine(location, *var);  // INTENT(INOUT)
          }
        }
      }
    }
  }

  void DoChecker::Leave(const parser::Expr &parsedExpr) {
    if (const SomeExpr * expr{GetExpr(parsedExpr)}) {
      ActualArgumentSet argSet{CollectActualArguments(*expr)};
      for (const evaluate::ActualArgumentRef &argRef : argSet) {
        CheckIfArgIsDoVar(*argRef, parsedExpr.source, context_);
      }
    }
  }
```

The function `CheckIfArgIsDoVar()` was shared with the checks for DO
variables being passed to SUBROUTINE calls.

At this point, my pull request was approved, and I merged it and deleted the
associated branch.
