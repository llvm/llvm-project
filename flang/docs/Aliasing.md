<!--===- docs/Aliasing.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# Aliasing in Fortran

```{contents}
---
local:
---
```

## Introduction

References to the ISO Fortran language standard here are given by subclause number
or constraint number and pertain to Fortran 2018.

## Dummy Arguments

### Basic rule

Fortran famously passes actual arguments by reference, and forbids callers
from associating multiple arguments on a call to conflicting storage when
doing so would cause the called subprogram to write to a bit of that
storage by means of one dummy argument and read or write that same bit
by means of another.
For example:
```
function f(a,b,j,k)
  real a(*), b(*)
  a(j) = 1.
  b(k) = 2.
  f = a(j) ! can optimize to: f = 1.
end function
```

This prohibition applies to programs (or programmers) and has been in place
since Fortran acquired subroutines and functions in Fortran II.

A Fortran compiler is free to assume that a program conforms with this rule
when optimizing; and while obvious violations should of course be diagnosed,
the programmer bears the responsibility to understand and comply with this rule.

It should be noted that this restriction on dummy argument aliasing works
"both ways", in general.
Modifications to a dummy argument cannot affect other names by which that
bit of storage may be known;
conversely, modifications to anything other than a dummy argument cannot
affect that dummy argument.

When a subprogram modifies storage by means of a particular dummy argument,
Fortran's prohibition against dummy argument aliasing is not limited just to other
dummy arguments, but to any other name by which that storage might be visible.
For example:
```
module m
  real x
 contains
  function f(y)
    real y
    x = 1.
    y = 2.
    f = x ! can optimize to: f = 1.
  end function
  subroutine bad
    print *, f(x) ! nonconforming usage!
  end subroutine
end module
```

Similar examples can be written using variables in `COMMON` blocks, host-association
in internal subprograms, and so forth.

Further, the general rule that a dummy argument by which some particular bit
of storage has been modified must be the only means by which that storage is
referenced during the lifetime of a subprogram extends to cover any associations
with that dummy argument via pointer association, argument association in
procedure references deeper on the call chain, and so on.

### Complications

Subclause 15.5.2.13 ("Restrictions on entities associated with dummy arguments"),
which the reader is encouraged to try to understand despite its opacity,
formalizes the rules for aliasing of dummy arguments.

In addition to the "basic rule" above, Fortran imposes these additional
requirements on programs.

1. When a dummy argument is `ALLOCATABLE` or `POINTER`, it can be deallocated
   or reallocated only through the dummy argument during the life of the
   subprogram.
1. When a dummy argument has a derived type with a component, possibly nested,
   that is `ALLOCATABLE` or `POINTER`, the same restriction applies.
1. If a subprogram ever deallocates or reallocates a dummy argument or one of its
   components, the program cannot access that data by any other means, even
   before the change in allocation.

That subclause also *relaxes* the rules against dummy argument aliasing in
some situations.

1. When a dummy argument is a `POINTER`, it is essentially treated like any
   other pointer for the purpose of alias analysis (see below), and its
   status as a dummy argument is reduced to being relevant only for
   deallocation and reallocation (see above).
1. When a dummy argument is a `TARGET`, the actual argument is really
   a variable (not an expression or something that needs to be passed via
   a temporary), and that variable could be a valid data target in a pointer
   assignment statement, then the compiler has to worry about aliasing
   between that dummy argument and pointers if some other circumstances
   apply.  (See the standard, this one is weird and complicated!)
1. Aliasing doesn't extend its restrictions to what other images might do
   to a coarray dummy argument's associated local storage during the lifetime
   of a subprogram -- i.e., other images don't have to worry about avoiding
   accesses to the local image's storage when its coarray nature is explicit
   in the declaration of the dummy argument.
   (But when the local image's storage is associated with a non-coarray dummy
   argument, the rules still apply.
   In other words, the compiler doesn't have to worry about corrays unless
   it sees coarrays.)

### Implications for inlining

A naive implementation of inlining might rewrite a procedure reference
something like this:
```
module m
 contains
  function addto(x, y)
    real, intent(in out) :: x
    real, intent(in) :: y
    x = x + y
    addto = y
  end function
  function f(a,j,k)
    real a(*)
    a(k) = 1.
    f = addto(a(j), a(k)) ! optimizable to 1.
  end function
end module
```

becoming, after inline expansion at the Fortran language level,

```
function f(a,j,k)
  real a(*)
  a(k) = 1.
  a(j) = a(j) + a(k)
  f = a(k) ! no longer optimizable to 1.
end function
```

The problem for a compiler is this: at the Fortran language level, no
language construct has the same useful guarantees against aliasing as
dummy arguments have.
A program transformation that changes dummy arguments into something
else needs to implement in its internal or intermediate representations
some kind of metadata that preserves assumptions against aliasing.

### `INTENT(IN)`

A dummy argument may have an`INTENT` attribute.
The relevant case for alias analysis is `INTENT(IN)`, as constraint
C844 prohibits the appearance of an `INTENT(IN)` non-pointer dummy
argument in any "variable definition context" (19.6.7), which is
Fortran's way of saying that it might be at risk of modification.

It would be great if the compiler could assume that an actual argument
that corresponds to an `INTENT(IN)` dummy argument is unchanged after
the called subprogram returns.
Unfortunately, the language has holes that admit ways by which an
`INTENT(IN)` dummy argument may change, even in a conforming program
(paragraph 2 and note 4 in subclause 8.5.10 notwithstanding).
In particular, Fortran nowhere states that a non-pointer `INTENT(IN)`
dummy argument is not "definable".

1. `INTENT(IN)` does not prevent the same variable from also being
   associated with another dummy argument in the same call *without*
   `INTENT(IN)` and being modified thereby, which is conforming so
   long as the subprogram never references the dummy argument that
   has `INTENT(IN)`.
   In other words, `INTENT(IN)` is necessary but not sufficient to
   guarantee safety from modification.
1. A dummy argument may have `INTENT(IN)` and `TARGET` attributes,
   and in a non-`PURE` subprogram this would allow modification of
   its effective argument by means of a local pointer.
1. An `INTENT(IN)` dummy argument may be forwarded to another
   procedure's dummy argument with no `INTENT` attribute, and is
   susceptible to being modified during that call.
   This case includes references to procedures with implicit
   interfaces.

So, for the purposes of use/def/kill analysis, associating a variable with
a non-`PURE` procedure's non-pointer dummy argument may be fraught
even when `INTENT(IN)` is present without `VALUE`.

Arguing the other side of this:
an interoperable procedure's `INTENT(IN)` dummy
arguments are forbidden from being modified, and it would be odd
for calls to foreign C functions to be safer than native calls (18.7).

### `VALUE`

A dummy argument with the `VALUE` attribute is effectively meant to
be copied into a temporary for a call and not copied back into
its original variable (if any).
A `VALUE` dummy argument is therefore as safe from aliasing as
a local variable of the subprogram is.

## Pointers and targets

Modern Fortran's pointers can't associate with arbitrary data.
They can be pointed only at objects that have the explicit `TARGET`
attribute, or at the targets of other pointers.

A variable that does not have the `TARGET` attribute is generally
safe from aliasing with pointers (but see exceptions below).
And generally, pointers must be assumed to alias all other pointers and
all `TARGET` data (perhaps reduced with data flow analysis).

A `VOLATILE` pointer can only point to a `VOLATILE` target, and
a non-`VOLATILE` pointer cannot.
A clever programmer might try to exploit this requirement to
clarify alias analysis, but I have not encountered such usage
so far.

### The `TARGET` hole for dummy arguments

An actual argument that doesn't have the `TARGET` attribute can still be
associated with a dummy argument that *is* a target.
This allows a non-target variable to become a target during the lifetime
of a call.
In a non-`PURE` subprogram (15.7), a pointer may be assigned to such a
dummy argument or to a portion of it.
Such a pointer has a valid lifetime that ends when the subprogram does.

### Valid lifetimes of pointers to dummy arguments

The Fortran standard doesn't mention compiler-generated and -populated
temporary storage in the context of argument association in 15.5.2,
apart from `VALUE`, but instead tries to name all of the circumstances
in which an actual argument's value may have to be transmitted by means
of a temporary in each of the paragraphs that constrain the usable
lifetimes of a pointer that has been pointed to a dummy argument
during a call.
It would be more clear, I think, had the standard simply described
the reasons for which an actual argument might have to occupy temporary
storage, and then just said that pointers to temporaries must not be
used once those temporaries no longer exist.

### Lack of pointer target `INTENT`

`INTENT` attributes for dummy pointer arguments apply to the pointer
itself, not to the data to which the pointer points.
Fortran still has no means of declaring a read-only pointer.
Fortran also has no rule against associating read-only data with a pointer.

### Cray pointers

Cray pointers are, or were, an extension that attempted to provide
some of the capabilities of modern pointers and allocatables before those
features were standardized.
They had some aliasing restrictions; in particular, Cray pointers were
not allowed to alias each other.

They are now more or less obsolete and we have no plan in place to
support them.

## Type considerations

Pointers with distinct types may alias so long as their types are
compatible in the sense of the standard.

Pointers to derived types and `COMPLEX` may alias with pointers to the
types of their components.
For example:
```
complex, pointer :: pz(:)
real, pointer :: pa(:)
pa => z(:)%re ! points to all of the real components
```

### Shape and rank

Array rank is not a material consideration to alias analysis.
Two pointers may alias even if their ranks or shapes differ.
For example, a pointer may associate with a column in a matrix
to which another pointer associates;
or a matrix pointer with only one column or one row may associate
with a vector.

It is also possible in Fortran to "remap" target data by establishing
a pointer of arbitrary rank as a view of a storage sequence.
For example:
```
real, target :: vector(100)
real, pointer :: matrix(:,:)
matrix(1:10,1:10) => v ! now vector's elements look like a matrix
```

## Selectors in `ASSOCIATE`, `SELECT TYPE`, and `CHANGE TEAM`

Selectors in `ASSOCIATE` and related constructs may associate with
either expression values or variables.
In the case of variables, the language imposes no restriction on
aliasing during the lifetime of the construct, and the compiler must
not assume that a selector works in a manner that is analogous to
that of a dummy argument.

## Allocatables

There really isn't anything special about `ALLOCATABLE` objects
from the perspective of aliasing, apart from rules (above) that requiring
`ALLOCATABLE` dummy arguments be (de)allocated only by way of the dummy
argument.

Because an `ALLOCATABLE` dummy argument preserves the values of lower
bounds and can be assumed to be contiguous, some programmers advocate
the use of explicitly `ALLOCATABLE` dummy arguments even when subprograms
do not modify their allocation status.
The usual aliasing restrictions still apply, even when the same `ALLOCATABLE`
is associated with two or more dummy arguments on a call.

## `ASYNCHRONOUS` and `VOLATILE`

These attributes can, unlike any other, be scoped in Fortran by means of
redeclaration in a `BLOCK` construct or nested procedure.

`ASYNCHRONOUS` data must be assumed to be read or written by some other
agent during its lifetime.
For example, Fortran's asynchronous I/O capabilities might be implemented
in a runtime support library by means of threading or explicitly asynchronous
system calls.
An MPI implementation might use `ASYNCHRONOUS` dummy arguments to indicate
that data transfers may take place during program execution in some way that
is not visible to the Fortran compiler.

The optimizer must handle `ASYNCHRONOUS` and `VOLATILE` data with great care.
Reads and writes of `ASYNCHRONOUS` data cannot be moved across statements
that might initiate or complete background operations.
Reads and writes of `VOLATILE` data should be treated like `volatile` in C:
there are no "dead" writes, reads cannot be CSE'd, and both operations should
be properly fenced.

## Storage assocation via `EQUIVALENCE`

A non-allocatable object, or parts of one, may have multiple names in Fortran
via `EQUIVALENCE`.
These objects cannot be have the `POINTER` or `TARGET` attributes.
Their offsets in static, stack, or `COMMON` storage is resolved by semantics
prior to lowering.

