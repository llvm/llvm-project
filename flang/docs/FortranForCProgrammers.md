<!--===- docs/FortranForCProgrammers.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# Fortran For C Programmers

```{contents}
---
local:
---
```

This note is limited to essential information about Fortran so that
a C or C++ programmer can get started more quickly with the language,
at least as a reader, and avoid some common pitfalls when starting
to write or modify Fortran code.
Please see other sources to learn about Fortran's rich history,
current applications, and modern best practices in new code.

## Know This At Least

* There have been many implementations of Fortran, often from competing
  vendors, and the standard language has been defined by U.S. and
  international standards organizations.  The various editions of
  the standard are known as the '66, '77, '90, '95, 2003, 2008, and
  (now) 2018 standards.
* Forward compatibility is important.  Fortran has outlasted many
  generations of computer systems hardware and software.  Standard
  compliance notwithstanding, Fortran programmers generally expect that
  code that has compiled successfully in the past will continue to
  compile and work indefinitely.  The standards sometimes designate
  features as being deprecated, obsolescent, or even deleted, but that
  can be read only as discouraging their use in new code -- they'll
  probably always work in any serious implementation.
* Fortran has two source forms, which are typically distinguished by
  filename suffixes.  `foo.f` is old-style "fixed-form" source, and
  `foo.f90` is new-style "free-form" source.  All language features
  are available in both source forms.  Neither form has reserved words
  in the sense that C does.  Spaces are not required between tokens
  in fixed form, and case is not significant in either form.
* Variable declarations are optional by default.  Variables whose
  names begin with the letters `I` through `N` are implicitly
  `INTEGER`, and others are implicitly `REAL`.  These implicit typing
  rules can be changed in the source.
* Fortran uses parentheses in both array references and function calls.
  All arrays must be declared as such; other names followed by parenthesized
  expressions are assumed to be function calls.
* Fortran has a _lot_ of built-in "intrinsic" functions.  They are always
  available without a need to declare or import them.  Their names reflect
  the implicit typing rules, so you will encounter names that have been
  modified so that they have the right type (e.g., `AIMAG` has a leading `A`
  so that it's `REAL` rather than `INTEGER`).
* The modern language has means for declaring types, data, and subprogram
  interfaces in compiled "modules", as well as legacy mechanisms for
  sharing data and interconnecting subprograms.

## A Rosetta Stone

Fortran's language standard and other documentation uses some terminology
in particular ways that might be unfamiliar.

| Fortran | English |
| ------- | ------- |
| Association | Making a name refer to something else |
| Assumed | Some attribute of an argument or interface that is not known until a call is made |
| Companion processor | A C compiler |
| Component | Class member |
| Deferred | Some attribute of a variable that is not known until an allocation or assignment |
| Derived type | C++ class |
| Dummy argument | C++ reference argument |
| Final procedure | C++ destructor |
| Generic | Overloaded function, resolved by actual arguments |
| Host procedure | The subprogram that contains a nested one |
| Implied DO | There's a loop inside a statement |
| Interface | Prototype |
| Internal I/O | `sscanf` and `snprintf` |
| Intrinsic | Built-in type or function |
| Polymorphic | Dynamically typed |
| Processor | Fortran compiler |
| Rank | Number of dimensions that an array has |
| `SAVE` attribute | Statically allocated |
| Type-bound procedure | Kind of a C++ member function but not really |
| Unformatted | Raw binary |

## Data Types

There are five built-in ("intrinsic") types: `INTEGER`, `REAL`, `COMPLEX`,
`LOGICAL`, and `CHARACTER`.
They are parameterized with "kind" values, which should be treated as
non-portable integer codes, although in practice today these are the
byte sizes of the data.
(For `COMPLEX`, the kind type parameter value is the byte size of one of the
two `REAL` components, or half of the total size.)
The legacy `DOUBLE PRECISION` intrinsic type is an alias for a kind of `REAL`
that should be more precise, and bigger, than the default `REAL`.

`COMPLEX` is a simple structure that comprises two `REAL` components.

`CHARACTER` data also have length, which may or may not be known at compilation
time.
`CHARACTER` variables are fixed-length strings and they get padded out
with space characters when not completely assigned.

User-defined ("derived") data types can be synthesized from the intrinsic
types and from previously-defined user types, much like a C `struct`.
Derived types can be parameterized with integer values that either have
to be constant at compilation time ("kind" parameters) or deferred to
execution ("len" parameters).

Derived types can inherit ("extend") from at most one other derived type.
They can have user-defined destructors (`FINAL` procedures).
They can specify default initial values for their components.
With some work, one can also specify a general constructor function,
since Fortran allows a generic interface to have the same name as that
of a derived type.

Last, there are "typeless" binary constants that can be used in a few
situations, like static data initialization or immediate conversion,
where type is not necessary.

## Arrays

Arrays are not types in Fortran.
Being an array is a property of an object or function, not of a type.
Unlike C, one cannot have an array of arrays or an array of pointers,
although can can have an array of a derived type that has arrays or
pointers as components.
Arrays are multidimensional, and the number of dimensions is called
the _rank_ of the array.
In storage, arrays are stored such that the last subscript has the
largest stride in memory, e.g. A(1,1) is followed by A(2,1), not A(1,2).
And yes, the default lower bound on each dimension is 1, not 0.

Expressions can manipulate arrays as multidimensional values, and
the compiler will create the necessary loops.

## Allocatables

Modern Fortran programs use `ALLOCATABLE` data extensively.
Such variables and derived type components are allocated dynamically.
They are automatically deallocated when they go out of scope, much
like C++'s `std::vector<>` class template instances are.
The array bounds, derived type `LEN` parameters, and even the
type of an allocatable can all be deferred to run time.
(If you really want to learn all about modern Fortran, I suggest
that you study everything that can be done with `ALLOCATABLE` data,
and follow up all the references that are made in the documentation
from the description of `ALLOCATABLE` to other topics; it's a feature
that interacts with much of the rest of the language.)

## I/O

Fortran's input/output features are built into the syntax of the language,
rather than being defined by library interfaces as in C and C++.
There are means for raw binary I/O and for "formatted" transfers to
character representations.
There are means for random-access I/O using fixed-size records as well as for
sequential I/O.
One can scan data from or format data into `CHARACTER` variables via
"internal" formatted I/O.
I/O from and to files uses a scheme of integer "unit" numbers that is
similar to the open file descriptors of UNIX; i.e., one opens a file
and assigns it a unit number, then uses that unit number in subsequent
`READ` and `WRITE` statements.

Formatted I/O relies on format specifications to map values to fields of
characters, similar to the format strings used with C's `printf` family
of standard library functions.
These format specifications can appear in `FORMAT` statements and
be referenced by their labels, in character literals directly in I/O
statements, or in character variables.

One can also use compiler-generated formatting in "list-directed" I/O,
in which the compiler derives reasonable default formats based on
data types.

## Subprograms

Fortran has both `FUNCTION` and `SUBROUTINE` subprograms.
They share the same name space, but functions cannot be called as
subroutines or vice versa.
Subroutines are called with the `CALL` statement, while functions are
invoked with function references in expressions.

There is one level of subprogram nesting.
A function, subroutine, or main program can have functions and subroutines
nested within it, but these "internal" procedures cannot themselves have
their own internal procedures.
As is the case with C++ lambda expressions, internal procedures can
reference names from their host subprograms.

## Modules

Modern Fortran has good support for separate compilation and namespace
management.
The *module* is the basic unit of compilation, although independent
subprograms still exist, of course, as well as the main program.
Modules define types, constants, interfaces, and nested
subprograms.

Objects from a module are made available for use in other compilation
units via the `USE` statement, which has options for limiting the objects
that are made available as well as for renaming them.
All references to objects in modules are done with direct names or
aliases that have been added to the local scope, as Fortran has no means
of qualifying references with module names.

## Arguments

Functions and subroutines have "dummy" arguments that are dynamically
associated with actual arguments during calls.
Essentially, all argument passing in Fortran is by reference, not value.
One may restrict access to argument data by declaring that dummy
arguments have `INTENT(IN)`, but that corresponds to the use of
a `const` reference in C++ and does not imply that the data are
copied; use `VALUE` for that.

When it is not possible to pass a reference to an object, or a sparse
regular array section of an object, as an actual argument, Fortran
compilers must allocate temporary space to hold the actual argument
across the call.
This is always guaranteed to happen when an actual argument is enclosed
in parentheses.

The compiler is free to assume that any aliasing between dummy arguments
and other data is safe.
In other words, if some object can be written to under one name, it's
never going to be read or written using some other name in that same
scope.
```
  SUBROUTINE FOO(X,Y,Z)
  X = 3.14159
  Y = 2.1828
  Z = 2 * X ! CAN BE FOLDED AT COMPILE TIME
  END
```
This is the opposite of the assumptions under which a C or C++ compiler must
labor when trying to optimize code with pointers.

## Overloading

Fortran supports a form of overloading via its interface feature.
By default, an interface is a means for specifying prototypes for a
set of subroutines and functions.
But when an interface is named, that name becomes a *generic* name
for its specific subprograms, and calls via the generic name are
mapped at compile time to one of the specific subprograms based
on the types, kinds, and ranks of the actual arguments.
A similar feature can be used for generic type-bound procedures.

This feature can be used to overload the built-in operators and some
I/O statements, too.

## Polymorphism

Fortran code can be written to accept data of some derived type or
any extension thereof using `CLASS`, deferring the actual type to
execution, rather than the usual `TYPE` syntax.
This is somewhat similar to the use of `virtual` functions in c++.

Fortran's `SELECT TYPE` construct is used to distinguish between
possible specific types dynamically, when necessary.  It's a
little like C++17's `std::visit()` on a discriminated union.

## Pointers

Pointers are objects in Fortran, not data types.
Pointers can point to data, arrays, and subprograms.
A pointer can only point to data that has the `TARGET` attribute.
Outside of the pointer assignment statement (`P=>X`) and some intrinsic
functions and cases with pointer dummy arguments, pointers are implicitly
dereferenced, and the use of their name is a reference to the data to which
they point instead.

Unlike C, a pointer cannot point to a pointer *per se*, nor can they be
used to implement a level of indirection to the management structure of
an allocatable.
If you assign to a Fortran pointer to make it point at another pointer,
you are making the pointer point to the data (if any) to which the other
pointer points.
Similarly, if you assign to a Fortran pointer to make it point to an allocatable,
you are making the pointer point to the current content of the allocatable,
not to the metadata that manages the allocatable.

Unlike allocatables, pointers do not deallocate their data when they go
out of scope.

A legacy feature, "Cray pointers", implements dynamic base addressing of
one variable using an address stored in another.

## Preprocessing

There is no standard preprocessing feature, but every real Fortran implementation
has some support for passing Fortran source code through a variant of
the standard C source preprocessor.
Since Fortran is very different from C at the lexical level (e.g., line
continuations, Hollerith literals, no reserved words, fixed form), using
a stock modern C preprocessor on Fortran source can be difficult.
Preprocessing behavior varies across implementations and one should not depend on
much portability.
Preprocessing is typically requested by the use of a capitalized filename
suffix (e.g., "foo.F90") or a compiler command line option.
(Since the F18 compiler always runs its built-in preprocessing stage,
no special option or filename suffix is required.)

## "Object Oriented" Programming

Fortran doesn't have member functions (or subroutines) in the sense
that C++ does, in which a function has immediate access to the members
of a specific instance of a derived type.
But Fortran does have an analog to C++'s `this` via *type-bound
procedures*.
This is a means of binding a particular subprogram name to a derived
type, possibly with aliasing, in such a way that the subprogram can
be called as if it were a component of the type (e.g., `X%F(Y)`)
and receive the object to the left of the `%` as an additional actual argument,
exactly as if the call had been written `F(X,Y)`.
The object is passed as the first argument by default, but that can be
changed; indeed, the same specific subprogram can be used for multiple
type-bound procedures by choosing different dummy arguments to serve as
the passed object.
The equivalent of a `static` member function is also available by saying
that no argument is to be associated with the object via `NOPASS`.

There's a lot more that can be said about type-bound procedures (e.g., how they
support overloading) but this should be enough to get you started with
the most common usage.

## Pitfalls

Variable initializers, e.g. `INTEGER :: J=123`, are _static_ initializers!
They imply that the variable is stored in static storage, not on the stack,
and the initialized value lasts only until the variable is assigned.
One must use an assignment statement to implement a dynamic initializer
that will apply to every fresh instance of the variable.
Be especially careful when using initializers in the newish `BLOCK` construct,
which perpetuates the interpretation as static data.
(Derived type component initializers, however, do work as expected.)

If you see an assignment to an array that's never been declared as such,
it's probably a definition of a *statement function*, which is like
a parameterized macro definition, e.g. `A(X)=SQRT(X)**3`.
In the original Fortran language, this was the only means for user
function definitions.
Today, of course, one should use an external or internal function instead.

Fortran expressions don't bind exactly like C's do.
Watch out for exponentiation with `**`, which of course C lacks; it
binds more tightly than negation does (e.g., `-2**2` is -4),
and it binds to the right, unlike what any other Fortran and most
C operators do; e.g., `2**2**3` is 256, not 64.
Logical values must be compared with special logical equivalence
relations (`.EQV.` and `.NEQV.`) rather than the usual equality
operators.

A Fortran compiler is allowed to short-circuit expression evaluation,
but not required to do so.
If one needs to protect a use of an `OPTIONAL` argument or possibly
disassociated pointer, use an `IF` statement, not a logical `.AND.`
operation.
In fact, Fortran can remove function calls from expressions if their
values are not required to determine the value of the expression's
result; e.g., if there is a `PRINT` statement in function `F`, it
may or may not be executed by the assignment statement `X=0*F()`.
(Well, it probably will be, in practice, but compilers always reserve
the right to optimize better.)

Unless they have an explicit suffix (`1.0_8`, `2.0_8`) or a `D`
exponent (`3.0D0`), real literal constants in Fortran have the
default `REAL` type -- *not* `double` as in the case in C and C++.
If you're not careful, you can lose precision at compilation time
from your constant values and never know it.
