<!--===- docs/DoConcurrent.md 
  
   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  
-->

# `DO CONCURRENT` isn't necessarily concurrent

```eval_rst
.. contents::
   :local:
```

A variant form of Fortran's primary looping construct was
added to the Fortran 2008 language standard with the apparent
intent of enabling more effective automatic parallel execution of code
written in the standard language without the use of
non-standard directives.
Spelled `DO CONCURRENT`, the construct takes a rectilinear iteration
space specification like `FORALL` and allows us to write
a multidimensional loop nest construct with a single `DO CONCURRENT`
statement and a single terminating `END DO` statement.

Within the body of a `DO CONCURRENT` loop the program must respect
a long list of restrictions on its use of Fortran language features.
Actions that obviously can't be executed in parallel or that
don't allow all iterations to execute are prohibited.
These include:
* Control flow statements that would prevent the loop nest from
  executing all its iterations: `RETURN`, `EXIT`, and any
  `GOTO` or `CYCLE` that leaves the construct.
* Image control statements: `STOP`, `SYNC`, `LOCK`/`UNLOCK`, `EVENT`,
  and `ALLOCATE`/`DEALLOCATE` of a coarray.
* Calling a procedure that is not `PURE`.
* Deallocation of any polymorphic entity, as that could cause
  an impure FINAL subroutine to be called.
* Messing with the IEEE floating-point control and status flags.
* Accepting some restrictions on data flow between iterations
  (i.e., none) and on liveness of modified objects after the loop.
  (The details are spelled out later.)

In return for accepting these restrictions, a `DO CONCURRENT` might
compile into code that exploits the parallel features of the target
machine to run the iterations of the `DO CONCURRENT` construct.
One needn't necessarily require OpenACC or OpenMP directives.

But it turns out that these rules, though *necessary* for safe parallel
execution, are not *sufficient*.
One may write conforming `DO CONCURRENT` constructs that cannot
be safely parallelized by a compiler; worse, one may write conforming
`DO CONCURRENT` constructs whose parallelizability a compiler cannot
determine even in principle -- forcing a conforming compiler to
assume the worst and generate sequential code.

## Localization

The Fortran language standard does not actually define `DO CONCURRENT` as a
concurrent construct, or even as a construct that imposes sufficient
requirements on the programmer to allow for parallel execution.
`DO CONCURRENT` is instead defined as executing the iterations
of the loop in some arbitrary order (see subclause 11.1.7.4.3 paragraph 3).

A `DO CONCURRENT` construct cannot modify an object in one iteration
and expect to be able to read it in another, or read it in one before it gets
modified by another -- there's no way to synchronize inter-iteration
communication with critical sections or atomics.

But a conforming `DO CONCURRENT` construct *can* modify an object in
multiple iterations of the loop so long as its only reads from that
object *after* having modified it earler in the *same* iteration.
(See 11.1.7.5 paragraph 4 for the details.)

For example:

```
  DO CONCURRENT (J=1:N)
    TMP = A(J) + B(J)
    C(J) = TMP
  END DO
  ! And TMP is undefined afterwards
```

The scalar variable `TMP` is used in this loop in a way that conforms
to the standard, as every use of `TMP` follows a definition that appears
earlier in the same iteration.

The idea, of course, is that a parallelizing compiler isn't required to
use the same word of memory to hold the value of `TMP`;
for parallel execution, `TMP` can be _localized_.
This means that the loop can be internally rewritten as if it had been
```
  DO CONCURRENT (J=1:N)
    BLOCK
      REAL :: TMP
      TMP = A(J) + B(J)
      C(J) = TMP
    END BLOCK
  END DO
```
and thus any risk of data flow between the iterations is removed.

## The identification problem

The automatic localization rules of `DO CONCURRENT` that allow
usage like `TMP` above are not limited to simple local scalar
variables.
They also apply to arbitrary variables, and thus may apply
in cases that a compiler cannot determine exactly due to
the presence of indexing, indirection, and interprocedural data flow.

Let's see why this turns out to be a problem.

Examples:
```
  DO CONCURRENT (J=1:N)
    T(IX(J)) = A(J) + B(J)
    C(J) = T(IY(J))
  END DO
```
This loop conforms to the standard language if,
whenever `IX(J)` equals `IY(J')` for any distinct pair of iterations
`J` and `J'`,
then the load must be reading a value stored earlier in the
same iteration -- so `IX(J')==IY(J')`, and hence `IX(J)==IX(J')` too,
in this example.
Otherwise, a load in one iteration might depend on a store
in another.

When all values of `IX(J)` are distinct, and the program conforms
to the restrictions of `DO CONCURRENT`, a compiler can parallelize
the construct easily without applying localization to `T(...)`.
And when some values of `IX(J)` are duplicates, a compiler can parallelize
the loop by forwarding the stored value to the load in those
iterations.
But at compilation time, there's _no way to distinguish_ these
cases in general, and a conservative implementation has to assume
the worst and run the loop's iterations serially.
(Or compare `IX(J)` with `IY(J)` at runtime and forward the
stored value conditionally, which adds overhead and becomes
quickly impractical in loops with multiple loads and stores.)

In
```
  TYPE :: T
    REAL, POINTER :: P
  END TYPE
  TYPE(T) :: T1(N), T2(N)
  DO CONCURRENT (J=1:N)
    T1(J)%P = A(J) + B(J)
    C(J) = T2(J)%P
  END DO
```
we have the same kind of ambiguity from the compiler's perspective.
Are the targets of the pointers used for the stores all distinct
from the targets of the pointers used for the loads?
The programmer may know that they are so, but a compiler
cannot; and there is no syntax by which one can stipulate
that they are so.

## The global variable localization problem

Here's another case:
```
  MODULE M
    REAL :: T
  END MODULE
  ...
  USE M
  INTERFACE
    PURE REAL FUNCTION F(X)
      REAL, INTENT(IN) :: X
    END FUNCTION
  END INTERFACE
  DO CONCURRENT (J=1:N)
    T = A(J) + B(J)
    D(J) = F(A(J)) + T
  END DO
```
The variable `T` is obviously meant to be localized.
However, a compiler can't be sure that the pure function `F`
doesn't read from `T`; if it does, there wouldn't be a
practical way to convey the localized copy to it.

In summary, standard Fortran defines `DO CONCURRENT` as a serial
construct with a sheaf of constraints that we assume are intended
to enable straightforward parallelization without
all of the complexity of defining threading models or shared memory semantics,
with the addition of an automatic localization rule that provides
convenient temporaries objects without requiring the use of nested
`BLOCK` or `ASSOCIATE` constructs.
But the language allows ambiguous cases in which a compiler can neither
1. prove that automatic localization *is* required for a given
   object in every iteration, nor
1. prove that automatic localization *isn't* required in any iteration.

## Locality specifiers

The Fortran 2018 standard added "locality specifiers" to the
`DO CONCURRENT` statement.
These allow one to define some variable names as being `LOCAL` or
`SHARED`, overriding the automatic localization rule so that it
applies only in the remaining cases of "unspecified" locality.

`LOCAL` variables are those that can be defined by more than one
iteration but are referenced only after having been defined
earlier in the same iteration.
`SHARED` variables are those that, if defined in
any iteration, are not defined or referenced in any other iteration.

(There is also a `LOCAL_INIT` specifier that is not relevant to the
problem at hand, and a `DEFAULT(NONE)` specifier that requires a
locality specifier be present for every variable mentioned in the
`DO CONCURRENT` construct.)

These locality specifiers can help resolve some otherwise ambiguous
cases of localization, but they're not a complete solution to the problems
described above.

First, the specifiers allow explicit localization of objects
(like the scalar `T` in `MODULE M` above) that are not local variables
of the subprogram.
`DO CONCURRENT` still allows a pure procedure called from the loop
to reference `T`, and so explicit localization just confirms the
worst-case assumptions about interprocedural data flow
within an iteration that a compiler must make anyway.

Second, the specifiers allow arbitary variables to be localized,
not just scalars.
One may localize a million-element array of derived type
with allocatable components to be created in each iteration,
for example.
(It is not clear whether localized objects are finalized;
probably not.)

Third, as Fortran uses context to distinguish references to
pointers from (de)references to their targets, it's not clear
whether `LOCAL(PTR)` localizes a pointer, its target, or both.

Fourth, the specifiers can be applied only to variable _names_,
not to any designator with subscripts or component references.
One may have defined a derived type to hold a representation
of a sparse matrix, using `ALLOCATABLE` components to store its
packed data and indexing structures, but a program cannot localize
some parts of it and share the rest.
(Perhaps one may wrap `ASSOCIATE` constructs around the
`DO CONCURRENT` construct;
the interaction between locality specifiers and construct entities is
not clearly defined in the language.)

In the example above that defines `T(IX(J))` and reads from `T(IY(J))`,
the locality specifiers can't be used to share those elements of `T()`
that are modified at most once and localize the cases where
`IX(J)` is a duplicate and `IY(J)==IX(J)`.

Last, when a loop both defines and references many shared objects,
including potential references to globally accessible object
in called procedures, one may need to name all of them in a `SHARED`
specifier.

## What to do now

These problems have been presented to the J3 Fortran language
standard committee.
Their responses in
recent [e-mail discussions](https://mailman.j3-fortran.org/pipermail/j3/2020-July/thread.html)
did not include an intent to address them in future standards or corrigenda.
The most effective-looking response -- which was essentially "just use
`DEFAULT(SHARED)` to disable all automatic localization" -- is not an
viable option, since the language does not include such a specifier!

Programmers writing `DO CONCURRENT` loops that are safely parallelizable
need an effective means to convey to compilers that those compilers
do not have to assume only the weaker stipulations required by
today's `DO CONCURRENT` without having to write verbose and
error-prone locality specifiers (when those would suffice).
Specifically, an easy means is required that stipulates that localization
should apply at most only to the obvious cases of local non-pointer
non-allocatable scalars.

In the LLVM Fortran compiler project (a/k/a "flang", "f18") we considered
several solutions to this problem.
1. Add syntax (e.g., `DO PARALLEL` or `DO CONCURRENT() DEFAULT(PARALLEL)`)
   by which one can inform the compiler that it should localize only
   the obvious cases of simple local scalars.
   Such syntax seems unlikely to ever be standardized, so its usage
   would be nonportable.
1. Add a command-line option &/or a source directive to stipulate
   the stronger guarantees.  Obvious non-parallelizable usage in the construct
   would elicit a stern warning.  The `DO CONCURRENT` loops in the source
   would continue to be portable to other compilers.
1. Assume that these stronger conditions hold by default, and add a command-line
   option &/or a source directive to "opt out" back to the weaker
   requirements of the standard language
   in the event that the program contains one of those inherently
   non-parallelizable `DO CONCURRENT` loops that perhaps should never have
   been possible to write in a conforming program in the first place.
   Actual parallel `DO CONCURRENT` constructs would produce parallel
   code for users who would otherwise be surprised to learn about these
   problems in the language.
   But this option could lead to non-standard behavior for codes that depend,
   accidentally or not, on non-parallelizable implicit localization.
1. Accept the standard as it exists, do the best job of automatic
   parallelization that can be done, and refer dissatisfied users to J3.
   This would be avoiding the problem.

None of these options is without a fairly obvious disadvantage.
The best option seems to be the one that assumes that users who write
`DO CONCURRENT` constructs are doing so with the intent to write parallel code.

## Other precedents

As of August 2020, we observe that the GNU Fortran compiler (10.1) does not
yet implement the Fortran 2018 locality clauses, but will parallelize some
`DO CONCURRENT` constructs without ambiguous data dependences when the automatic
parallelization option is enabled.

The Intel Fortran compiler supports the new locality clauses and will parallelize
some `DO CONCURRENT` constructs when automatic parallelization option is enabled.
When OpenMP is enabled, ifort reports that all `DO CONCURRENT` constructs are
parallelized, but they seem to execute in a serial fashion when data flow
hazards are present.
