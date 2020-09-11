<!--===- docs/Calls.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# Representation of Fortran function calls

```eval_rst
.. contents::
   :local:
```

## Procedure reference implementation protocol

Fortran function and subroutine references are complicated.
This document attempts to collect the requirements imposed by the 2018
standard (and legacy extensions) on programs and implementations, work
through the implications of the various features, and propose both a
runtime model and a compiler design.

All section, requirement, and constraint numbers herein pertain to
the Fortran 2018 standard.

This note does not consider calls to intrinsic procedures, statement
functions, or calls to internal runtime support library routines.

## Quick review of terminology

* A *dummy argument* is a function or subroutine parameter.
  It is *associated* with an *effective argument* at each call
  to the procedure.
* The *shape* of an array is a vector containing its extent (size)
  on each dimension; the *rank* of an array is the number of its
  dimensions (i.e., the shape of its shape).
  The absolute values of the lower and upper bounds of the dimensions
  of an array are not part of its shape, just their difference (plus 1).
* An *explicit-shape* array has all of its bounds specified; lower
  bounds default to 1.  These can be passed by with a single address
  and their contents are contiguous.
* An *assumed-size* array is an explicit-shape array with `*` as its
  final dimension, which is the most-significant one in Fortran
  and whose value does not affect indexed address calculations.
* A *deferred-shape* array (`DIMENSION::A(:)`) is a `POINTER` or `ALLOCATABLE`.
  `POINTER` target data might not be contiguous.
* An *assumed-shape* (not size!) array (`DIMENSION::A(:)`) is a dummy argument
  that is neither `POINTER` nor `ALLOCATABLE`; its lower bounds can be set
  by the procedure that receives them (defaulting to 1), and its
  upper bounds are functions of the lower bounds and the extents of
  dimensions in the *shape* of the effective argument.
* An *assumed-length* `CHARACTER(*)` dummy argument
  takes its length from the effective argument.
* An *assumed-length* `CHARACTER(*)` *result* of an external function (C721)
  has its length determined by its eventual declaration in a calling scope.
* An *assumed-rank* `DIMENSION::A(..)` dummy argument array has an unknown
  number of dimensions.
* A *polymorphic* `CLASS(t)` dummy argument, `ALLOCATABLE`, or `POINTER`
  has a specific derived type or some extension of that type.
  An *unlimited polymorphic* `CLASS(*)` object can have any
  intrinsic or derived type.
* *Interoperable* `BIND(C)` procedures are written in C or callable from C.

## Interfaces

Referenced procedures may or may not have declared interfaces
available to their call sites.

Procedures with some post-Fortran '77 features *require* an
explicit interface to be called (15.4.2.2) or even passed (4.3.4(5)):

* use of argument keywords in a call
* procedures that are `ELEMENTAL` or `BIND(C)`
* procedures that are required to be `PURE` due to the context of the call
  (specification expression, `DO CONCURRENT`, `FORALL`)
* dummy arguments with these attributes: `ALLOCATABLE`, `POINTER`,
  `VALUE`, `TARGET`, `OPTIONAL`, `ASYNCHRONOUS`, `VOLATILE`,
  and, as a consequence of limitations on its use, `CONTIGUOUS`;
  `INTENT()`, however, does *not* require an explicit interface
* dummy arguments that are coarrays
* dummy arguments that are assumed-shape or assumed-rank arrays
* dummy arguments with parameterized derived types
* dummy arguments that are polymorphic
* function result that is an array
* function result that is `ALLOCATABLE` or `POINTER`
* `CHARACTER` function result whose length is neither constant
  nor assumed
* derived type function result with `LEN` type parameter value that is
  not constant
  (note that result derived type parameters cannot be assumed (C795))

Module procedures, internal procedures, procedure pointers,
type-bound procedures, and recursive references by a procedure to itself
always have explicit interfaces.
(Consequently, they cannot be assumed-length `CHARACTER(*)` functions;
conveniently, assumed-length `CHARACTER(*)` functions are prohibited from
recursion (15.6.2.1(3))).

Other uses of procedures besides calls may also require explicit interfaces,
such as procedure pointer assignment, type-bound procedure bindings, &c.

Note that non-parameterized monomorphic derived type arguments do
*not* by themselves require the use of an explicit interface.
However, dummy arguments with any derived type parameters *do*
require an explicit interface, even if they are all `KIND` type
parameters.

15.5.2.9(2) explicitly allows an assumed-length `CHARACTER(*)` function
to be passed as an actual argument to an explicit-length dummy;
this has implications for calls to character-valued dummy functions
and function pointers.
(In the scopes that reference `CHARACTER` functions, they must have
visible definitions with explicit result lengths.)

### Implicit interfaces

In the absence of any characteristic or context that *requires* an
explicit interface (see above), an external function or subroutine (R503)
or `ENTRY` (R1541) can be called directly or indirectly via its implicit interface.
Each of the arguments can be passed as a simple address, including
dummy procedures.
Procedures that *can* be called via an implicit interface can
undergo more thorough checking
by semantics when an explicit interface for them exists, but they must be
compiled as if all calls to them were through the implicit interface.
This note will mention special handling for procedures that are exposed
to the possibility of being called with an implicit interface as *F77ish* procedures
below; this is of course not standard terminology.

Internal and module subprograms that are ever passed as arguments &/or
assigned as targets of procedure pointers may be F77ish.

Every F77ish procedure can and must be distiguished at compilation time.
Such procedures should respect the external naming conventions (when external)
and any legacy ABI used for Fortran '77 programs on the target architecture,
so that portable libraries can be compiled
and used by distinct implementations (and their versions)
of Fortran.

Note that F77ish functions still have known result types, possibly by means
of implicit typing of their names.
They can also be `CHARACTER(*)` assumed-length character functions.

In other words: these F77sh procedures that do not require the use of an explicit
interface and that can possibly be referenced, directly or indirectly,
with implicit interfaces are limited to argument lists that comprise
only the addresses of effective arguments and the length of a `CHARACTER` function result
(when there is one), and they can return only scalar values with constant
type parameter values.
None of their arguments or results need be (or can be) implemented
with descriptors,
and any internal procedures passed to them as arguments must be
simple addresses of non-internal subprograms or trampolines for
internal procedures.

Note that the `INTENT` attribute does not, by itself,
require the use of explicit interface; neither does the use of a dummy
procedure (implicit or explicit in their interfaces).
So the analyis of calls to F77ish procedures must allow for the
invisible use of `INTENT(OUT)`.

## Protocol overview

Here is a summary script of all of the actions that may need to be taken
by the calling procedure and its referenced procedure to effect
the call, entry, exit, and return steps of the procedure reference
protocol.
The order of these steps is not particularly strict, and we have
some design alternatives that are explored further below.

### Before the call:

1. Compute &/or copy into temporary storage the values of
   some effective argument expressions and designators (see below).
1. Create and populate descriptors for arguments that use them
   (see below).
1. Possibly allocate function result storage,
   when its size can be known by all callers; function results that are
   neither `POINTER` nor `ALLOCATABLE` must have explicit shapes (C816).
1. Create and populate a descriptor for the function result, if it
   needs one (deferred-shape/-length `POINTER`, any `ALLOCATABLE`,
   derived type with non-constant length parameters, &c.).
1. Capture the values of host-escaping local objects in memory;
   package them into single address (for calls to internal procedures &
   for calls that pass internal procedures as arguments).
1. Resolve the target procedure's polymorphic binding, if any.
1. Marshal effective argument addresses (or values for `%VAL()` and some
   discretionary `VALUE` arguments) into registers.
1. Marshal `CHARACTER` argument lengths in additional value arguments for
   `CHARACTER` effective arguments not passed via descriptors.
   These lengths must be 64-bit integers.
1. Marshal an extra argument for the length of a `CHARACTER` function
   result if the function is F77ish.
1. Marshal an extra argument for the function result's descriptor,
   if it needs one.
1. Set the "host instance" (static link) register when calling an internal
   procedure from its host or another internal procedure, a procedure pointer,
   or dummy procedure (when it has a descriptor).
1. Jump.

### On entry:
1. For subprograms with alternate `ENTRY` points: shuffle `ENTRY` dummy arguments
   set a compiler-generated variable to identify the alternate entry point,
   and jump to the common entry point for common processing and a `switch()`
   to the statement after the `ENTRY`.
1. Capture `CHARACTER` argument &/or assumed-length result length values.
1. Complete `VALUE` copying if this step will not always be done
   by the caller (as I think it should be).
1. Finalize &/or re-initialize `INTENT(OUT)` non-pointer
   effective arguments (see below).
1. For interoperable procedures called from C: compact discontiguous
   dummy argument values when necessary (`CONTIGUOUS` &/or
   explicit-shape/assumed-size arrays of assumed-length `CHARACTER(*)`).
1. Optionally compact assumed-shape arguments for contiguity on one
   or more leading dimensions to improve SIMD vectorization, if not
   `TARGET` and not already sufficiently contiguous.
   (PGI does this in the caller, whether the callee needs it or not.)
1. Complete allocation of function result storage, if that has
   not been done by the caller.
1. Initialize components of derived type local variables,
   including the function result.

Execute the callee, populating the function result or selecting
the subroutine's alternate return.

### On exit:
1. Clean up local scope (finalization, deallocation)
1. Deallocate `VALUE` argument temporaries.
   (But don't finalize them; see 7.5.6.3(3)).
1. Replace any assumed-shape argument data that were compacted on
   entry for contiguity when the data were possibly
   modified across the call (never when `INTENT(IN)` or `VALUE`).
1. Identify alternate `RETURN` to caller.
1. Marshal results.
1. Jump

### On return to the caller:
1. Save the result registers, if any.
1. Copy effective argument array designator data that was copied into
   a temporary back into its original storage (see below).
1. Complete deallocation of effective argument temporaries (not `VALUE`).
1. Reload definable host-escaping local objects from memory, if they
   were saved to memory by the host before the call.
1. `GO TO` alternate return, if any.
1. Use the function result in an expression.
1. Eventually, finalize &/or deallocate the function result.

(I've omitted some obvious steps, like preserving/restoring callee-saved
registers on entry/exit, dealing with caller-saved registers before/after
calls, and architecture-dependent ABI requirements.)

## The messy details

### Copying effective argument values into temporary storage

There are several conditions that require the compiler to generate
code that allocates and populates temporary storage for an actual
argument.

First, effective arguments that are expressions, not designators, obviously
need to be computed and captured into memory in order to be passed
by reference.
This includes parenthesized designators like `(X)`, which are
expressions in Fortran, as an important special case.
(This case also technically includes unparenthesized constants,
but those are better implemented by passing addresses in read-only
memory.)
The dummy argument cannot be known to have `INTENT(OUT)` or
`INTENT(IN OUT)`.

Small scalar or elemental `VALUE` arguments may be passed in registers,
as should arguments wrapped in the legacy VMS `%VAL()` notation.
Multiple elemental `VALUE` arguments might be packed into SIMD registers.

Effective arguments that are designators, not expressions, must also
be copied into temporaries in the following situations.

1. Coindexed objects need to be copied into the local image.
   This can get very involved if they contain `ALLOCATABLE`
   components, which also need to be copied, along with their
   `ALLOCATABLE` components, and may be best implemented with a runtime
   library routine working off a description of the type.
1. Effective arguments associated with dummies with the `VALUE`
   attribute need to be copied; this can be done on either
   side of the call, but there are optimization opportunities
   available when the caller's side bears the responsibility.
1. In non-elemental calls, the values of array sections with
   vector-valued subscripts need to be gathered into temporaries.
   These effective arguments are not definable, and they are not allowed to
   be associated with non-`VALUE` dummy arguments with the attributes
   `INTENT(OUT)`, `INTENT(IN OUT)`, `ASYNCHRONOUS`, or `VOLATILE`
   (15.5.2.4(21)); `INTENT()` can't always be checked.
1. Non-simply-contiguous (9.5.4) arrays being passed to non-`POINTER`
   dummy arguments that must be contiguous (due to a `CONTIGUOUS`
   attribute, or not being assumed-shape or assumed-rank; this
   is always the case for F77ish procedures).
   This should be a runtime decision, so that effective arguments
   that turn out to be contiguous can be passed cheaply.
   This rule does not apply to coarray dummies, whose effective arguments
   are required to be simply contiguous when this rule would otherwise
   force the use of a temporary (15.5.2.8); neither does it apply
   to `ASYNCHRONOUS` and `VOLATILE` effective arguments, which are
   disallowed when copies would be necessary (C1538 - C1540).
   *Only temporaries created by this contiguity requirement are
   candidates for being copied back to the original variable after
   the call* (see below).

Fortran requires (18.3.6(5)) that calls to interoperable procedures
with dummy argument arrays with contiguity requirements
handle the compaction of discontiguous data *in the Fortran callee*,
at least when called from C.
And discontiguous data must be compacted on the *caller's* side
when passed from Fortran to C (18.3.6(6)).

We could perform all argument compaction (discretionary or
required) in the callee, but there are many cases where the
compiler knows that the effective argument data are contiguous
when compiling the caller (a temporary is needed for other reasons,
or the effective argument is simply contiguous) and a run-time test for
discontiguity in the callee can be avoided by using a caller-compaction
convention when we have the freedom to choose.

While we are unlikely to want to _needlessly_ use a temporary for
an effective argument that does not require one for any of these
reasons above, we are specifically disallowed from doing so
by the standard in cases where pointers to the original target
data are required to be valid across the call (15.5.2.4(9-10)).
In particular, compaction of assumed-shape arrays for discretionary
contiguity on the leading dimension to ease SIMD vectorization
cannot be done safely for `TARGET` dummies without `VALUE`.

Effective arguments associated with known `INTENT(OUT)` dummies that
require allocation of a temporary -- and this can only be for reasons of
contiguity -- don't have to populate it, but they do have to perform
minimal initialization of any `ALLOCATABLE` components so that
the runtime doesn't crash when the callee finalizes and deallocates
them.
`ALLOCATABLE` coarrays are prohibited from being affected by `INTENT(OUT)`
(see C846).
Note that calls to implicit interfaces must conservatively allow
for the use of `INTENT(OUT)` by the callee.

Except for `VALUE` and known `INTENT(IN)` dummy arguments, the original
contents of local designators that have been compacted into temporaries
could optionally have their `ALLOCATABLE` components invalidated
across the call as an aid to debugging.

Except for `VALUE` and known `INTENT(IN)` dummy arguments, the contents of
the temporary storage will be copied back into the effective argument
designator after control returns from the procedure, and it may be necessary
to preserve addresses (or the values of subscripts and cosubscripts
needed to recalculate them) of the effective argument designator, or its
elements, in additional temporary storage if they can't be safely or
quickly recomputed after the call.

### `INTENT(OUT)` preparation

Effective arguments that are associated with `INTENT(OUT)`
dummy arguments are required to be definable.
This cannot always be checked, as the use of `INTENT(OUT)`
does not by itself mandate the use of an explicit interface.

`INTENT(OUT)` arguments are finalized (as if) on entry to the called
procedure.  In particular, in calls to elemental procedures,
the elements of an array are finalized by a scalar or elemental
`FINAL` procedure (7.5.6.3(7)).

Derived type components that are `ALLOCATABLE` are finalized
and deallocated; they are prohibited from being coarrays.
Components with initializers are (re)initialized.

The preparation of effective arguments for `INTENT(OUT)` could be
done on either side of the call.  If the preparation is
done by the caller, there is an optimization opportunity
in situations where unmodified incoming `INTENT(OUT)` dummy
arguments whose types lack `FINAL` procedures are being passed
onward as outgoing `INTENT(OUT)` arguments.

### Arguments and function results requiring descriptors

Dummy arguments are represented with the addresses of new descriptors
when they have any of the following characteristics:

1. assumed-shape array (`DIMENSION::A(:)`)
1. assumed-rank array (`DIMENSION::A(..)`)
1. parameterized derived type with assumed `LEN` parameters
1. polymorphic (`CLASS(T)`, `CLASS(*)`)
1. assumed-type (`TYPE(*)`)
1. coarray dummy argument
1. `INTENT(IN) POINTER` argument (15.5.2.7, C.10.4)

`ALLOCATABLE` and other `POINTER` arguments can be passed by simple
address.

Non-F77ish procedures use descriptors to represent two further
kinds of dummy arguments:

1. assumed-length `CHARACTER(*)`
1. dummy procedures

F77ish procedures use other means to convey character length and host instance
links (respectively) for these arguments.

Function results are described by the caller & callee in
a caller-supplied descriptor when they have any of the following
characteristics, some which necessitate an explicit interface:

1. deferred-shape array (so `ALLOCATABLE` or `POINTER`)
1. derived type with any non-constant `LEN` parameter
   (C795 prohibit assumed lengths)
1. procedure pointer result (when the interface must be explicit)

Storage for a function call's result is allocated by the caller when
possible: the result is neither `ALLOCATABLE` nor `POINTER`,
the shape is scalar or explicit, and the type has `LEN` parameters
that are constant expressions.
In other words, the result doesn't require the use of a descriptor
but can't be returned in registers.
This allows a function result to be written directly into a local
variable or temporary when it is safe to treat the variable as if
it were an additional `INTENT(OUT)` argument.
(Storage for `CHARACTER` results, assumed or explicit, is always
allocated by the caller, and the length is always passed so that
an assumed-length external function will work when eventually
called from a scope that declares the length that it will use
(15.5.2.9 (2)).)

Note that the lower bounds of the dimensions of non-`POINTER`
non-`ALLOCATABLE` dummy argument arrays are determined by the
callee, not the caller.
(A Fortran pitfall: declaring `A(0:9)`, passing it to a dummy
array `D(:)`, and assuming that `LBOUND(D,1)` will be zero
in the callee.)
If the declaration of an assumed-shape dummy argument array
contains an explicit lower bound expression (R819), its value
needs to be computed by the callee;
it may be captured and saved in the incoming descriptor
as long as we assume that argument descriptors can be modified
by callees.
Callers should fill in all of the fields of outgoing
non-`POINTER` non-`ALLOCATABLE` argument
descriptors with the assumption that the callee will use 1 for
lower bound values, and callees can rely on them being 1 if
not modified.

### Copying temporary storage back into argument designators

Except for `VALUE` and known `INTENT(IN)` dummy arguments and array sections
with vector-valued subscripts (15.5.2.4(21)), temporary storage into
which effective argument data were compacted for contiguity before the call
must be redistributed back to its original storage by the caller after
the return.

In conjunction with saved cosubscript values, a standard descriptor
would suffice to represent a pointer to the original storage into which the
temporary data should be redistributed;
the descriptor need not be fully populated with type information.

Note that coindexed objects with `ALLOCATABLE` ultimate components
are required to be associated only with dummy arguments with the
`VALUE` &/or `INTENT(IN)` attributes (15.6.2.4(6)), so there is no
requirement that the local image somehow reallocate remote storage
when copying the data back.

### Polymorphic bindings

Calls to the type-bound procedures of monomorphic types are
resolved at compilation time, as are calls to `NON_OVERRIDABLE`
type-bound procedures.
The resolution of calls to overridable type-bound procedures of
polymorphic types must be completed at execution (generic resolution
of type-bound procedure bindings from effective argument types, kinds,
and ranks is always a compilation-time task (15.5.6, C.10.6)).

Each derived type that declares or inherits any overridable
type-bound procedure bindings must correspond to a static constant
table of code addresses (or, more likely, a static constant type
description containing or pointing to such a table, along with
information used by the runtime support library for initialization,
copying, finalization, and I/O of type instances).  Each overridable
type-bound procedure in the type corresponds to an index into this table.

### Host instance linkage

Calls to dummy procedures and procedure pointers that resolve to
internal procedures need to pass an additional "host instance" argument that
addresses a block of storage in the stack frame of the their
host subprogram that was active at the time they were passed as an
effective argument or associated with a procedure pointer.
This is similar to a static link in implementations of programming
languages with nested subprograms, although Fortran only allows
one level of nesting.
The 64-bit x86 and little-endian OpenPower ABIs reserve registers
for this purpose (`%r10` & `R11`); 64-bit ARM has a reserved register
that can be used (`x18`).

The host subprogram objects that are visible to any of their internal
subprograms need to be resident in memory across any calls to them
(direct or not).  Any host subprogram object that might be defined
during a call to an internal subprogram needs to be reloaded after
a call or reside permanently in memory.
A simple conservative analysis of the internal subprograms can
identify all of these escaping objects and their definable subset.

The address of the host subprogram storage used to hold the escaping
objects needs to be saved alongside the code address(es) that
represent a procedure pointer.
It also needs to be conveyed alongside the text address for a
dummy procedure.

For F77ish procedures, we cannot use a "procedure pointer descriptor"
to pass a procedure argument -- they expect to receive a single
address argument.
We will need to package the host instance link in a trampoline
that loads its address into the designated register.

GNU Fortran and Intel Fortran construct trampolines by writing
a sequence of machine instructions to a block of storage in the
host's stack frame, which requires the stack to be executable,
which seems inadvisable for security reasons;
XLF manages trampolines in its runtime support library, which adds some overhead
to their construction and a reclamation obligation;
NAG Fortran manages a static fixed-sized stack of trampolines
per call site, imposing a hidden limit on recursion and foregoing
reentrancy;
PGI passes host instance links in descriptors in additional arguments
that are not always successfully forwarded across implicit interfaces,
sometimes leading to crashes when they turn out to be needed.

F18 will manage a pool of trampolines in its runtime support library
that can be used to pass internal procedures as effective arguments
to F77ish procedures, so that
a bare code address can serve to represent the effective argument.
But targets that can only be called with an explicit interface
have the option of using a "fat pointer" (or additional argument)
to represent a dummy procedure closure so as
to avoid the overhead of constructing and reclaiming a trampoline.
Procedure descriptors can also support multiple code addresses.

### Naming

External subroutines and functions (R503) and `ENTRY` points (R1541)
with `BIND(C)` (R808) have linker-visible names that are either explicitly
specified in the program or determined by straightforward rules.
The names of other F77ish external procedures should respect the conventions
of the target architecture for legacy Fortran '77 programs; this is typically
something like `foo_`.

In other cases, however, we have fewer constraints on external naming,
as well as some additional requirements and goals.

Module procedures need to be distinguished by the name of their module
and (when they have one) the submodule where their interface was
defined.
Note that submodule names are distinct in their modules, not hierarchical,
so at most two levels of qualification are needed.

Pure `ELEMENTAL` functions (15.8) must use distinct names for any alternate
entry points used for packed SIMD arguments of various widths if we support
calls to these functions in SIMD parallel contexts.
There are already conventions for these names in `libpgmath`.

The names of non-F77ish external procedures
should be distinguished as such so that incorrect attempts to call or pass
them with an implicit interface will fail to resolve at link time.
Fortran 2018 explicitly enables us to do this with a correction to Fortran
2003 in 4.3.4(5).

Last, there must be reasonably permanent naming conventions used
by the F18 runtime library for those unrestricted specific intrinsic
functions (table 16.2 in 16.8) and extensions that can be passed as
arguments.

In these cases where external naming is at the discretion
of the implementation, we should use names that are not in the C language
user namespace, begin with something that identifies
the current incompatible version of F18, the module, the submodule, and
elemental SIMD width, and are followed by the external name.
The parts of the external name can be separated by some character that
is acceptable for use in LLVM IR and assembly language but not in user
Fortran or C code, or by switching case
(so long as there's a way to cope with extension names that don't begin
with letters).

In particular, the period (`.`) seems safe to use as a separator character,
so a `Fa.` prefix can serve to isolate these discretionary names from
other uses and to identify the earliest link-compatible version.
For examples: `Fa.mod.foo`, `Fa.mod.submod.foo`, and (for an external
subprogram that requires an explicit interface) `Fa.foo`.
When the ABI changes in the future in an incompatible way, the
initial prefix becomes `Fb.`, `Fc.`, &c.

## Summary of checks to be enforced in semantics analysis

8.5.10 `INTENT` attributes
* (C846) An `INTENT(OUT)` argument shall not be associated with an
  object that is or has an allocatable coarray.
* (C847) An `INTENT(OUT)` argument shall not have `LOCK_TYPE` or `EVENT_TYPE`.

8.5.18 `VALUE` attribute
* (C863) The argument cannot be assumed-size, a coarray, or have a coarray
  ultimate component.
* (C864) The argument cannot be `ALLOCATABLE`, `POINTER`, `INTENT(OUT)`,
  `INTENT(IN OUT)`, or `VOLATILE`.
* (C865) If the procedure is `BIND(C)`, the argument cannot be `OPTIONAL`.

15.5.1 procedure references:
* (C1533) can't pass non-intrinsic `ELEMENTAL` as argument
* (C1536) alternate return labels must be in the inclusive scope
* (C1537) coindexed argument cannot have a `POINTER` ultimate component

15.5.2.4 requirements for non-`POINTER` non-`ALLOCATABLE` dummies:
* (2) dummy must be monomorphic for coindexed polymorphic actual
* (2) dummy must be polymorphic for assumed-size polymorphic actual
* (2) dummy cannot be `TYPE(*)` if effective is PDT or has TBPs or `FINAL`
* (4) character length of effective cannot be less than dummy
* (6) coindexed effective with `ALLOCATABLE` ultimate component requires
      `INTENT(IN)` &/or `VALUE` dummy
* (13) a coindexed scalar effective requires a scalar dummy
* (14) a non-conindexed scalar effective usually requires a scalar dummy,
  but there are some exceptions that allow elements of storage sequences
  to be passed and treated like explicit-shape or assumed-size arrays
  (see 15.5.2.11)
* (16) array rank agreement
* (20) `INTENT(OUT)` & `INTENT(IN OUT)` dummies require definable actuals
* (21) array sections with vector subscripts can't be passed to definable dummies
       (`INTENT(OUT)`, `INTENT(IN OUT)`, `ASYNCHRONOUS`, `VOLATILE`)
* (22) `VOLATILE` attributes must match when dummy has a coarray ultimate component
* (C1538 - C1540) checks for `ASYNCHRONOUS` and `VOLATILE`

15.5.2.5 requirements for `ALLOCATABLE` & `POINTER` arguments when both
the dummy and effective arguments have the same attributes:
* (2) both or neither can be polymorphic
* (2) both are unlimited polymorphic or both have the same declared type
* (3) rank compatibility
* (4) effective argument must have deferred the same type parameters as the dummy

15.5.2.6 `ALLOCATABLE` dummy arguments:
* (2) effective must be `ALLOCATABLE`
* (3) corank must match
* (4) coindexed effective requires `INTENT(IN)` dummy
* (7) `INTENT(OUT)` & `INTENT(IN OUT)` dummies require definable actuals

15.5.2.7 `POINTER` dummy arguments:
* (C1541) `CONTIGUOUS` dummy requires simply contiguous actual
* (C1542) effective argument cannot be coindexed unless procedure is intrinsic
* (2) effective argument must be `POINTER` unless dummy is `INTENT(IN)` and
  effective could be the right-hand side of a pointer assignment statement

15.5.2.8 corray dummy arguments:
* (1) effective argument must be coarray
* (1) `VOLATILE` attributes must match
* (2) explicitly or implicitly contiguous dummy array requires a simply contiguous actual

15.5.2.9 dummy procedures:
* (1) explicit dummy procedure interface must have same characteristics as actual
* (5) dummy procedure `POINTER` requirements on effective arguments

15.6.2.1 procedure definitions:
* `NON_RECURSIVE` procedures cannot recurse.
* Assumed-length `CHARACTER(*)` functions cannot be declared as `RECURSIVE`, array-valued,
  `POINTER`, `ELEMENTAL`, or `PURE' (C723), and cannot be called recursively (15.6.2.1(3)).
* (C823) A function result cannot be a coarray or contain a coarray ultimate component.

`PURE` requirements (15.7): C1583 - C1599.
These also apply to `ELEMENTAL` procedures that are not `IMPURE`.

`ELEMENTAL` requirements (15.8.1): C15100-C15103,
and C1533 (can't pass as effective argument unless intrinsic)

For interoperable procedures and interfaces (18.3.6):
* C1552 - C1559
* function result is scalar and of interoperable type (C1553, 18.3.1-3)
* `VALUE` arguments are scalar and of interoperable type
* `POINTER` dummies cannot be `CONTIGUOUS` (18.3.6 paragraph 2(5))
* assumed-type dummies cannot be `ALLOCATABLE`, `POINTER`, assumed-shape, or assumed-rank (18.3.6 paragraph 2 (5))
* `CHARACTER` dummies that are `ALLOCATABLE` or `POINTER` must be deferred-length

## Further topics to document

* Alternate return specifiers
* `%VAL()`, `%REF()`, and `%DESCR()` legacy VMS interoperability extensions
* Unrestricted specific intrinsic functions as effective arguments
* SIMD variants of `ELEMENTAL` procedures (& unrestricted specific intrinsics)
* Elemental subroutine calls with array arguments
