## Concept
The properties that characterize data values and objects in Fortran
programs must sometimes be materialized when the program runs.

Some properties are known during compilation and constant during
execution, yet must be reified anyway for execution in order to
drive the interfaces of a language support library or the mandated
interfaces of interoperable (i.e., C) procedure calls.

Note that many Fortran intrinsic subprograms have interfaces
that are more flexible and generic than actual Fortran subprograms
can be, so properties that must be known during compilation and
are constant during execution may still need to be materialized
for calls to the library, even if only by modifying names to
distinguish types or their kind specializations.

Other properties are deferred to execution, and need to be represented
to serve the needs of compiled code and the run time support library.

Previous implementations of Fortran have typically defined a small
sheaf of _descriptor_ data structures for this purpose, and attached
these descriptors as additional hidden arguments, type components,
and local variables so as to convey dynamic characteristics between
subprograms and between user code and the run-time support library.

### References
References are to the 12-2017 draft of the Fortran 2018 standard
(N2146).

Section 15.4.2.2 can be interpreted as a decent list of things that
might need descriptors or other hidden state passed across a
subprogram call, since such features (apart from assumed-length
`CHARACTER` function results) trigger a requirement for the
subprogram to have an explicit interface visible to their callers.

Section 15.5.2 has good laundry lists of situations that can arise
across subprogram call boundaries.

## A survey of dynamic characteristics

### Length of assumed-length `CHARACTER` function results (B.3.6)
```
CHARACTER*8 :: FOO
PRINT *, FOO('abcdefghijklmnopqrstuvwxyz')
...
CHARACTER*(*) FUNCTION FOO(STR)
  CHARACTER*26 STR
  FOO=STR
END
```

prints `abcdefgh` because the length parameter of the character type
of the result of `FOO` is passed across the call -- even in the absence
of an explicit interface!

### Assumed length type parameters (7.2)
Dummy arguments and associate names for `SELECT TYPE` can have assumed length
type parameters, which are denoted by asterisks (not colons).
Their values come from actual arguments or the associated expression (resp.).

### Explicit-shape arrays (8.5.8.2)
The expressions used for lower and upper bounds must be captured and remain
invariant over the scope of an array, even if they contain references to
variables that are later modified.

Explicit-shape arrays can be dummy arguments, "adjustable" local variables,
and components of derived type (using specification expressions in terms
of constants and KIND type parameters).

### Leading dimensions of assumed-size arrays (8.5.8.5)
```
SUBROUTINE BAR(A)
  REAL A(2,3,*)
END
```
The total size and final dimension's extent do not constitute dynamic
properties.
The called subprogram has no means to extract the extent of the
last (major) dimension, and may not depend upon it implicitly by using
the array in any context that demands a known shape.

The values of the expressions used as the bounds of the dimensions
that appear prior to
the last dimension are, however, effectively captured on entry to the
subprogram, and remain invariant even if the variables that appear in
those expressions have their values modified later.
This is similar to the requirements for an explicit-shape array.

### Some function results
1. Deferred-shape
2. Deferred length type parameter values
3. Stride information for `POINTER` results

Note that while function result variables can have the `ALLOCATABLE`
attribute, the function itself and the value returned to the caller
do not possess the attribute.

### Assumed-shape arrays
The extents of the dimensions of assumed-shape dummy argument arrays
are conveyed from those of the actual effective arguments.
The bounds, however, are not.  The called subprogram can define the
lower bound to be a value other than 1, but that is a local effect
only.

### Deferred-shape arrays
The extents and bounds of `POINTER` and `ALLOCATABLE` arrays are
established by pointer assignments and `ALLOCATE` statements.
Note that dummy arguments and function results that are `POINTER`
or `ALLOCATABLE` can be deferred-shape, not assumed-shape -- one cannot
supply a lower bound expression as a local effect.

### Strides
Some arrays can have discontiguous (or negative) strides.
These include assumed-shape dummy arguments and deferred-shape
`POINTER` variables, components, and function results.

Fortran disallows some conceivable cases that might otherwise
require implied strides, such as passing an array of an extended
derived type as an actual argument that corresponds to a
nonpolymorphic dummy array of a base type, or the similar
case of pointer assignment to a base of an extended derived type.

Other arrays, including `ALLOCATABLE`, can be assured to
be contiguous, and do not necessarily need to manage or
convey dynamic stride information.
`CONTIGUOUS` dummy arguments and `POINTER` arrays need not
record stride information either.
(The standard notes that a `CONTIGUOUS POINTER` occupies a
number of storage units that is distinct from that required
to hold a non-`CONTIGUOUS` pointer.)

Note that Fortran distinguishes the `CONTIGUOUS` attribute from
the concept of being known or required to be _simply contiguous_ (9.5.4),
which includes `CONTIGUOUS` entities as well as many others, and
the concept of actually _being_ contiguous (8.5.7) during execution.
I believe that the property of being simply contiguous implies
that an entity is known at compilation time to not require the
use or maintenance of hidden stride values.

### Derived type component initializers
Fortran allows components of derived types to be declared with
initial values that are to be assigned to the components when an
instance of the derived type is created.
These include `ALLOCATABLE` components, which are always initialized
to a deallocated state.

These can be implemented with constructor subroutines, inline
stores or block copies from static initializer blocks, or a sequence
of sparse offset/size/value component initializers to be emplaced
by the run-time library.

N.B. Fortran allows kind type parameters to appear in component
initialization constant expressions, but not length type parameters,
so the initialization values are constants.

N.B. Initialization is not assignment, and cannot be implemented
with assignments to uninitialized derived type instances from
static constant initializers.

### Polymorphic `CLASS()`, `CLASS(*)`, and `TYPE(*)`
Type identification for `SELECT TYPE`.
Default initializers (see above).
Offset locations of `ALLOCATABLE` and polymorphic components.
Presence of `FINAL` procedures.
Mappings to overridable type-bound specific procedures.

### Deferred length type parameters
Derived types with length type parameters, and `CHARACTER`, may be used
with the values of those parameters deferred to execution.
Their actual values must be maintained as characteristics of the dynamic
type that is associated with a value or object
.
A single copy of the deferred length type parameters suffices for
all of the elements of an array of that parameterized derived type.

### Components whose types and/or shape depends on length type parameters
Non-pointer, non-allocatable components whose types or shapes are expressed
in terms of length type parameters will probably have to be implemented as
if they had deferred type and/or shape and were `ALLOCATABLE`.
The derived type instance constructor must allocate them and possibly
initialize them; the instance destructor must deallocate them.

### Assumed rank arrays
Rank is almost always known at compilation time and would be redundant
in most circumstances if also managed dynamically.
`DIMENSION(..)` dummy arguments (8.5.8.7), however, are a recent feature
with which the rank of a whole array is dynamic outside the cases of
a `SELECT RANK` construct.

The lower bounds of the dimensions of assumed rank arrays
are always 1.

### Cached invariant subexpressions for addressing
Implementations of Fortran have often maintained precalculated integer
values to accelerate subscript computations.
For example, given `REAL*8 :: A(2:4,3:5)`, the data reference `A(I,J)`
resolves to something like `&A + 8*((I-2)+3*(J-3))`, and this can be
effectively reassociated to `&A - 88 + 8*I + 24*J`
or `&A - 88 + 8*(I + 3*J)`.
When the offset term and coefficients are not compile-time constants,
they are at least invariant and can be precomputed.

In the cases of dummy argument arrays, `POINTER`, and `ALLOCATABLE`,
these addressing invariants could be managed alongside other dynamic
information like deferred extents and lower bounds to avoid their
recalculation.
It's not clear that it's worth the trouble to do so, since the
expressions are invariant and cheap.

### Coarray state (8.5.6)
A _coarray_ is an `ALLOCATABLE` variable or component, or statically
allocated variable (`SAVE` attribute explicit or implied), or dummy
argument whose ultimate effective argument is one of such things.

Each image in a team maintains its portion of each coarray and can
access those portions of the coarray that are maintained by other images
in the team.
Allocations and deallocations are synchronization events at which
the several images can exchange whatever information is needed by
the underlying intercommunication interface to access the data
of their peers.
(Strictly speaking, an implementation could synchronize
images at allocations and deallocations with simple barriers, and defer
the communication of remote access information until it is needed for a
given coarray on a given image, so long as it could be acquired in a
"one-sided" fashion.)

### Presence of `OPTIONAL` dummy arguments
Typically indicated with null argument addresses.
Note that `POINTER` and `ALLOCATABLE` objects can be passed to
non-`POINTER` non-`ALLOCATABLE` dummy arguments, and their
association or allocation status (resp.) determines the presence
of the dummy argument.

### Stronger contiguity enforcement or indication
Some implementations of Fortran guarantee that dummy argument arrays
are, or have been made to be, contiguous on one or more dimensions
when the language does not require them to be so (8.5.7 p2).
Others pass a flag to identify contiguous arrays (or could pass the
number of contiguous leading dimensions, although I know of no such
implementation) so that optimizing transformations that depend on
contiguity can be made conditional with multiple-version code generation
and selected during execution.

In the absence of a contiguity guarantee or flag, the called side
would have to determine contiguity dynamically, if it cares,
by calculating addresses of elements in the array whose subscripts
differ by exactly 1 on exactly 1 dimension of interest, and checking
whether that difference exactly matches the byte size of the type times
the product of the extents of any prior dimensions.

### Host instances for dummy procedures and procedure pointers
A static link or other means of accessing the imported state of the
host procedure must be available when an internal procedure is
used as an actual argument or as a pointer assignment target.

### Alternate returns
Subroutines (only) with alternate return arguments need a
means, such as the otherwise unused function return value, by which
to distinguish and identify the use of an alternate `RETURN` statement.
The protocol can be a simple nonzero integer that drives a switch
in the caller, or the caller can pass multiple return addresses as
arguments for the callee to substitute on the stack for the original
return address in the event of an alternate `RETURN`.

## Implementation options

### A note on array descriptions
Some arrays require dynamic management of distinct combinations of
values per dimension.

One can extract the extent on a dimension from its bounds, or extract
the upper bound from the extent and the lower bound.  Having distinct
extent and upper bound would be redundant.

Contiguous arrays can assume a stride of 1 on each dimension.

Assumed-shape and assumed-size dummy argument arrays need not convey
lower bounds.

So there are examples of dimensions with
 * extent only (== upper bound): `CONTIGUOUS` assumed-shape, explict shape and multidimensional assumed-size with constant lower bound
 * lower bound and either extent or upper bound: `ALLOCATABLE`, `CONTIGUOUS` `POINTER`, general explicit-shape and multidimensional assumed-size
 * extent (== upper bound) and stride: general (non-`CONTIGUOUS`) assumed-shape
 * lower bound, stride, and either extent or upper bound: general (non-`CONTIGUOUS`) `POINTER`, assumed-rank

and these cases could be accompanied by precomputed invariant
addressing subexpressions to accelerate indexing calculations.

### Interoperability requirements

Fortran 2018 requires that a Fortran implementation supply a header file
`ISO_Fortran_binding.h` for use in C and C++ programs that defines and
implements an interface to Fortran objects from the _interoperable_
subset of Fortran objects and their types suitable for use when those
objects are passed to C functions.
This interface mandates a fat descriptor that is passed by address,
containing (at least)
 * a data base address
 * explicit rank and type
 * flags to distinguish `POINTER` and `ALLOCATABLE`
 * elemental byte size, and
 * (per-dimension) lower bound, extent, and byte stride

The requirements on the interoperability API do not mandate any
support for features like derived type component initialization,
automatic deallocation of `ALLOCATABLE` components, finalization,
derived type parameters, data contiguity flags, &c.
But neither does the Standard preclude inclusion of additional
interfaces to describe and support such things.

Given a desire to fully support the Fortran 2018 language, we need
to either support the interoperability requirements as a distinct
specialization of the procedure call protocol, or use the
`ISO_Fortran_binding.h` header file requirements as a subset basis for a
complete implementation that adds representations for all the
missing capabilities, which would be isolated and named so as
to prevent user C code from relying upon them.

### Design space
There is a range of possible options for representing the
properties of values and objects during the execution of Fortran
programs.

At one extreme, the amount of dynamic information is minimized,
and is packaged in custom data structures or additional arguments
for each situation to convey only the values that are unknown at
compilation time and actually needed at execution time.

At the other extreme, data values and objects are described completely,
including even the values of properties are known at compilation time.
This is not as silly as it sounds -- e.g., Fortran array descriptors
have historically materialized the number of dimensions they cover, even
though rank will be (nearly) always be a known constant during compilation.

When data are packaged, their containers can be self-describing to
some degree.
Description records can have tag values or strings.
Their fields can have presence flags or identifying tags, and fields
need not have fixed offsets or ordering.
This flexibility can increase binary compatibility across revisions
of the run-time support library, and is convenient for debugging
that library.
However, it is not free.

Further, the requirements of the representation of dynamic
properties of values and objects depend on the execution model:
specifically, are the complicated semantics of intrinsic assignment,
deallocation, and finalization of allocatables implemented entirely
in the support library, in generated code for non-recursive cases,
or by means of a combination of the two approaches?

Consider how to implement the following:
```
TYPE :: LIST
  REAL :: HEAD
  TYPE(LIST), ALLOCATABLE :: REST
END TYPE LIST
TYPE(LIST), ALLOCATABLE :: A, B
...
A = B
```

Fortran requires that `A`'s arbitrary-length linked list be deleted and
replaced with a "deep copy" of `B`'s.
So either a complicated pair of loops must be generated by the compiler,
or a sophisticated run time support library needs to be driven with
an expressive representation of type information.

## Proposal
We need to write `ISO_Fortran_binding.h` in any event.
It is a header that is published for use in user C code for interoperation
with compiled Fortran and the Fortran run time support library.

There is a sole descriptor structure defined in `ISO_Fortran_binding.h`.
It is suitable for characterizing scalars and array sections of intrinsic
types.
It is essentially a "fat" data pointer that encapsulates a raw data pointer,
a type code, rank, elemental byte size, and per-dimension bounds and stride.

Please note that the mandated interoperable descriptor includes the data
pointer.
This design in the Standard precludes the use of static descriptors that
could be associated with dynamic base addresses.

The F18 runtime cannot use just the mandated interoperable
`struct CFI_cdesc_t` argument descriptor structure as its
all-purpose data descriptor.
It has no information about derived type components, overridable
type-bound procedure bindings, type parameters, &c.

However, we could extend the standard interoperable argument descriptor.
The `struct CFI_cdesc_t` structure is not of fixed size, but we
can efficiently locate the first address after an instance of the
standard descriptor and attach our own data record there to
hold what we need.
There's at least one unused padding byte in the standard argument
descriptor that can be used to hold a flag indicating the presence
of the addenda.

The definitions of our additional run time data structures must
appear in a header file that is distinct from `ISO_Fortran_binding.h`,
and they should never be used by user applications.

This expanded descriptor structure can serve, at least initially for
simplicity, as the sole representation of `POINTER` variables and
components, `ALLOCATABLE` variables and components, and derived type
instances, including length parameter values.

An immediate concern with this concept is the amount of space and
initialization time that would be wasted when derived type components
needing a descriptor would have to be accompanied by an instance
of the general descriptor.
(In the linked list example close above, what could be done with a
single pointer for the `REST` component would become at least
a four-word dynamic structure.)
This concern is amplified when derived type instances
are allocated as arrays, since the overhead is per-element.

We can reduce this wastage in two ways.
First, when the content of the component's descriptor is constant
at compilation apart from its base address, a static descriptor
can be placed in read-only storage and attached to the description
of the derived type's components.
Second, we could eventually optimize the storage requirements by
omitting all static fields from the dynamic descriptor, and
expand the compressed dynamic descriptor during execution when
needed.
