The approach of FIR and lowering design so far was to start with the minimal set
of IR operations that could allow implementing the core aspects of Fortran (like
memory allocations, array addressing, runtime descriptors, and structured
control flow operations). One notable aspect of the current FIR is that array
and character operations are buffered (some storage is allocated for the result,
and the storage is addressed to implement the operation).  While this proved
functional so far, the code lowering expressions and assignments from the
front-end representations (the evaluate::Expr and parser nodes) to FIR has
significantly grown in complexity while it still lacks some F95 features around
character array expressions or FORALL. This is mainly explained by the fact that
the representation level gap is big, and a lot is happening in lowering.  It
appears more and more that some intermediate steps would help to split concerns
between translating the front-end representation to MLIR, implementing some
Fortran concepts at a lower-level (like character or derived type assignments),
and how bufferizations of character and array expressions should be done.

This document proposes the addition of two concepts and a set of related
operations in a new dialect HLFIR to allow a simpler lowering to a higher-level
FIR representation that would later be lowered to the current FIR representation
via MLIR translation passes.  As a result of these additions, it is likely that
the fir.array_load/fir.array_merge_store and related array operations could be
removed from FIR since array assignment analysis could directly happen on the
higher-level FIR representation.


The main principles of the new lowering design are:
-   Make expression lowering context independent and rather naive
-   Do not materialize temporaries while lowering to FIR
-   Preserve Fortran semantics/information for high-level optimizations

The core impact on lowering will be:
-   Lowering expressions and assignments in the exact same way, regardless of
    whether it is an array assignment context and/or an expression inside a
    forall.
-   Lowering transformational intrinsics in a verbatim way (no runtime calls and
    memory aspects yet).
-   Lowering character expressions in a verbatim way (no memcpy/runtime calls
    and memory aspects yet).
-   Argument association side effects will be delayed (copy-in/copy-out) to help
    inlining/function specialization to get rid of them when they are not
    relevant.


## Variable and Expression value concepts in HLFIR

## Strengthening the variable concept

Fortran variables are currently represented in FIR as mlir::Value with reference
or box type coming from special operations or block arguments. They are either
the result of a fir.alloca, fir.allocmem, or fir.address_of operations with the
mangled name of the variable as attribute, or they are function block arguments
with the mangled name of the variable as attribute.

Fortran variables are defined with a Fortran type (both dynamic and static) that
may have type parameters, a rank and shape (including lower bounds), and some
attributes (like TARGET, OPTIONAL, VOLATILE...). All this information is
currently not represented in FIR. Instead, lowering keeps track of all this
information in the fir::ExtendedValue lowering data structure and uses it when
needed. If unused in lowering, some information about variables is lost (like
non-constant array bound expressions). In the IR, only the static type, the
compile time constant extents, and compile time character lengths can be
retrieved from the mlir::Value of a variable in the general case (more can be
retrieved if the variable is tracked via a fir.box, but not if it is a bare
memory reference).

This makes reasoning about Fortran variables in FIR harder, and in general
forces lowering to apply all decisions related to the information that is lost
in FIR. A more problematic point is that it does not allow generating debug
information for the variables from FIR, since the bounds and type parameters
information is not tightly linked to the base mlir::Value.

The proposal is to add a hlfir.declare operation that would anchor the
fir::ExtendedValue information in the IR. A variable will be represented by a
single SSA value with a memory type (fir.ref<T>, fir.ptr<T>, fir.heap<T>,
fir.box<T>, fir.boxchar or fir.ref<fir.box<T>>). Not all memory types will be
allowed for a variable: it should allow retrieving all the shape, type
parameters, and dynamic type information without requiring to look-up for any
defining operations. For instance, `fir.ref<fir.array<?xf32>>` will not be
allowed as an HLFIR variable, and fir.box<fir.array<?xf32>> will be used
instead. However, `fir.ref<fir.array<100xf32>>` will be allowed to represent an
array whose lower bounds are all ones (if the lower bounds are different than
one, a fir.box will still be needed).  The hlfir.declare operation will be
responsible for producing the SSA value with the right memory type given the
variable specifications. One notable point is that, except for the POINTER and
ALLOCATABLE attributes that are retrievable from the SSA value type, other
Fortran attributes (OPTIONAL, TARGET, VOLATILE...) will not be retrievable from
the SSA value alone, and it will be required to access the defining
hlfir.declare to get the full picture.

This means that semantically relevant attributes will need to be set by
lowering on operations using variables when that is relevant (for instance when
using an OPTIONAL variable in an intrinsic where it is allowed to be absent).
Then, the optimizations passes will be allowed to look for the defining
hlfir.declare, but not to assume that it must be visible.  For instance, simple
contiguity of fir.box can be deduced in certain case from the hlfir.declare,
and if the hlfir.declare cannot be found, transformation passes will have to
assume that the variable may be non-contiguous.

In practice, it is expected that the passes will be able to leverage
hlfir.declare in most cases, but that guaranteeing that it will always be the
case would constraint the IR and optimizations too much.  The goal is also to
remove the fir.box usages when possible while lowering to FIR, to avoid
needlessly creating runtime descriptors for variables that do not really
require it.

The hlfir.declare operation and restrained memory types will allow:
- Pushing higher-level Fortran concepts into FIR operations (like array
  assignments or transformational intrinsics).
- Generating debug information for the variables based on the hlfir.declare
  operation.
- Generic Fortran aliasing analysis (currently implemented only around array
  assignments with the fir.array_load concept).

The hlfir.declare will have a sibling fir.declare operation in FIR that will
allow keeping variable information until debug info is generated. The main
difference is that the fir.declare will simply return its first operand,
making its codegen a no-op, while hlfir.declare might change the type of
its first operand to return an HLFIR variable compatible type.
The fir.declare op is the only operation described by this change that will be
added to FIR. The rational for this is that it is intended to survive until
LLVM dialect codegeneration so that debug info generation can use them and
alias information can take advantage of them even on FIR.

Note that Fortran variables are not necessarily named objects, they can also be
the result of function references returning POINTERs. hlfir.declare will also
accept such variables to be described in the IR (a unique name will be built
from the caller scope name and the function name.). In general, fir.declare
will allow to view every memory storage as a variable, and this will be used to
describe and use compiler created array temporaries.

## Adding an expression value concept in HLFIR

Currently, Fortran expressions can be represented as SSA values for scalar
logical, integer, real, and complex expressions. Scalar character or
derived-type expressions and all array expressions are buffered in lowering:
their results are directly given a memory storage in lowering and are
manipulated as variables.

While this keeps FIR simple, this makes the amount of IR generated for these
expressions higher, and in general makes later optimization passes job harder
since they present non-trivial patterns (with memory operations) and cannot be
eliminated by naive dead code elimination when the result is unused. This also
forces lowering to combine elemental array expressions into single loop nests to
avoid bufferizing all array sub-expressions (which would yield terrible
performance). These combinations, which are implemented using C++ lambdas in
lowering makes lowering code harder to understand. It also makes the expression
lowering code context dependent (especially designators lowering). The lowering
code paths may be different when lowering a syntactically similar expression in
an elemental expression context, in a forall context, or in a normal context.

Some of the combinations described in [Array Composition](ArrayComposition.md)
are currently not implemented in lowering because they are less trivial
optimizations, and do not really belong in lowering. However, deploying such
combinations on the generated FIR with bufferizations requires the usage of
non-trivial pattern matching and rewrites (recognizing temporary allocation,
usage, and related runtime calls). Note that the goal of such combination is not
only about inlining transformational runtime calls, it is mainly about never
generating a temporary for an array expression sub-operand that is a
transformational intrinsic call matching certain criteria. So the optimization
pass will not only need to recognize the intrinsic call, it must understand the
context it is being called in.

The usage of memory manipulations also makes some of the alias analysis more
complex, especially when dealing with foralls (the alias analysis cannot simply
follow an operand tree, it must understand indirect dependencies from operations
stored in memory).

The proposal is to add a !hlfir.expr<T> SSA value type concept, and set of
character operations (concatenation, TRIM, MAX, MIN, comparisons...), a set of
array transformational operations (SUM, MATMUL, TRANSPOSE, ...), and a generic
hlfir.elemental operation. The hlfir.expr<T> type is not intended to be used
with scalar types that already have SSA value types (e.g., integer or real
scalars).  Instead, these existing SSA types will implicitly be considered as
being expressions when used in high-level FIR operations, which will simplify
interfacing with other dialects that define operations with these types (e.g.,
the arith dialect).

These hlfir.expr values could then be placed in memory when needed (assigned to
a variable, passed as a procedure argument, or an IO output item...) via
hlfir.assign or hlfir.associate operations that will later be described.

When no special optimization pass is run, a translation pass would lower the
operations producing hlfir.expr to buffer allocations and memory operations just
as in the currently generated FIR.

However, these high-level operations should allow the writing of optimization
passes combining chains of operations producing hlfir.expr into optimized forms
via pattern matching on the operand tree.

The hlfir.elemental operation will be discussed in more detail below. It allows
simplifying lowering while keeping the ability to combine elemental
sub-expressions into a single loop nest. It should also allow rewriting some of
the transformational intrinsic operations to functions of the indices as
described in [Array Composition](ArrayComposition.md).

## Proposed design for HLFIR (High-Level Fortran IR)

### HLFIR Operations and Types

#### Introduce a hlfir.expr<T> type

Motivation: avoid the need to materialize expressions in temporaries while
lowering.

Syntax: ``` !hlfir.expr<[extent x]* T [, class]> ```

- `[extent x]*` represents the shape for arrays similarly to !fir.array<> type,
  except that the shape cannot be assumed rank (!hlfir.expr<..xT> is invalid).
  This restriction can be added because it is impossible to create an assumed
  rank expression in Fortran that is not a variable.
- `T` is the element type of the static type
- `class` flag can be set to denote that this a polymorphic expression (that the
  dynamic type should not be assumed to be the static type).


examples: !hlfir.expr<fir.char<?>>, !hlfir.expr<10xi32>,
!hlfir.expr<?x10x?xfir.complex<4>>

T in scalar hlfir.expr<T> can be:
-   A character type (fir.char<10, kind>, fir.char<?, kind>)
-   A derived type: (fir.type<t{...}>)

T in an array hlfir.expr< e1 x ex2 ..  : T> can be:
-   A character or derived type
-   A logical type (fir.logical<kind>)
-   An integer type (i1, i32, ….)
-   A floating point type (f32, f16…)
-   A complex type (fir.complex<4> or mlir::complex<f32>...)

Some expressions may be polymorphic (for instance, MERGE can be used on
polymorphic entities). The hlfir.expr type has an optional "class" flag to
denote this: hlfir.expr<T, class>.

Note that the ALLOCATABLE, POINTER, TARGET, VOLATILE, ASYNCHRONOUS, OPTIONAL
aspects do not apply to expressions, they apply to variables.

It is possible to query the following about an expression:
-   What is the extent : via hlfir.get_extent %expr, dim
-   What are the length parameters: via hlfir.get_typeparam %expr [, param_name]
-   What is the dynamic type: via hlfir.get_dynamic_type %expr

It is possible to get the value of an array expression element:
- %element = hlfir.apply %expr, %i, %j : (!hlfir.expr<T>, index index) ->
  hlfir.expr<ScalarType> | AnyConstantSizeScalarType

It is not directly possible to take an address for the expression, but an
expression value can be associated to a new variable whose address can be used
(required when passing the expression in a user call, or to concepts that are
kept low level in FIR, like IO runtime calls).  The variable created may be a
compiler created temporary, or may relate to a Fortran source variable if this
mechanism is used to implement ASSOCIATE.

-   %var = hlfir.associate %expr [attributes about the association]->
    AnyMemoryOrBoxType
-   hlfir.end_association %var

The intention is that the hlfir.expr<T> is the result of an operation, and
should most often not be a block argument. This is because the hlfir.expr is
mostly intended to allow combining chains of operations into more optimal
forms. But it is possible to represent any expression result via a Fortran
runtime descriptor (fir.box<T>), implying that if a hlfir.expr<T> is passed as
a block argument, the expression bufferization pass will evaluate the operation
producing the expression in a temporary, and transform the block operand into a
fir.box describing the temporary. Clean-up for the temporary will be inserted
after the last use of the hlfir.expr. Note that, at least at first, lowering
may help FIR to find the last use of a hlfir.expr by explicitly inserting a
hlfir.finalize %expr operation that may turn into a no-op if the expression is
not later materialized in memory.

It is nonetheless not intended that such abstract types be used as block
arguments to avoid introducing allocations and descriptor manipulations.

#### hlfir.declare operation

Motivation: represent variables, linking together a memory storage, shape,
length parameters, attributes and the variable name.

Syntax:
```
%var = hlfir.declare %base [shape %extent1, %extent2, ...] [lbs %lb1, %lb2, ...] [typeparams %l1, ...] {fir.def = mangled_variable_name, attributes} : [(....) ->] T1, T2
```

%var#0 will have a FIR memory type that is allowed for HLFIR variables. %var#1
will have the same type as %base, it is intended to be used when lowering HLFIR
to FIR in order to avoid creating unnecessary fir.box (that would become
runtime descriptors). When an HLFIR operation has access to the defining
hlfir.declare of its variable operands, the operation codegen will be allowed
to replace the %var#0 reference by the simpler %var#1 reference.

- Extents should only be provided if %base is not a fir.box and the entity is an
  array.
- lower bounds should only be provided if the entity is an array and the lower
  bounds are not default (all ones). It should also not be provided for POINTERs
  and ALLOCATABLES since the lower bounds may change.
- type parameters should be provided for entities with length parameters, unless
  the entity is a CHARACTER where the length is constant in %base type.
- The attributes will include the Fortran attributes: TARGET (fir.target),
  POINTER (fir.ptr), ALLOCATABLE (fir.alloc), CONTIGUOUS (fir.contiguous),
  OPTIONAL (fir.optional), VOLATILE (fir.volatile), ASYNCHRONOUS (fir.async).
  They will also indicate when an entity is part of an equivalence by giving the
  equivalence name (fir.equiv = mangled_equivalence_name).

hlfir.declare will be used for all Fortran variables, except the ones created via
the ASSOCIATE construct that will use hlfir.associate described below.

hlfir.declare will also be used when creating compiler created temporaries, in
which case the fir.tmp attribute will be given.

Examples:

| FORTRAN                                   | HLFIR                                                                                                                                                                                                                    |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| REAL :: X                                 | %mem = fir.alloca f32 <br> %x = hlfir.declare %mem {fir.def = "\_QPfooEx"} : fir.ref<f32>, fir.ref<f32>                                                                                                                  |
| REAL, TARGET :: X(10)                     | %mem = fir.alloca f32 <br> %nval = fir.load %n <br> %x = hlfir.declare %mem {fir.def = "\_QPfooEx", fir.target} : fir.ref<fir.array<10xf32>>, fir.ref<fir.array<10xf32>>                                                 |
| REAL :: X(N)                              | %mem = // … alloc or dummy argument <br> %nval = fir.load %n : i64 <br> %x = hlfir.declare %mem shape %nval {fir.def = "\_QPfooEx"} : (i64) -> fir.box<fir.array<?xf32>>, fir.ref<fir.array<?xf32>>                      |
| REAL :: X(0:)                             | %mem = // … dummy argument <br> %c0 = arith.constant 0 : index <br> %x = hlfir.declare %mem lbs %c0 {fir.def = "\_QPfooEx"} : (index) -> fir.box<fir.array<?xf32>>, fir.box<fir.array<?xf32>>                            |
| <br>REAL, POINTER :: X(:)                 | %mem = // … dummy argument, or local, or global <br> %x = hlfir.declare %mem {fir.def = "\_QPfooEx", fir.ptr} :  fir.ref<fir.box<fir.ptr<fir.array<?xf32>>>>, fir.ref<fir.box<fir.ptr<fir.array<?xf32>>>>                |
| REAL, ALLOCATABLE :: X(:)                 | %mem = // … dummy argument, or local, or global <br> %x = hlfir.declare %mem {fir.def = "\_QPfooEx", fir.alloc} :  fir.ref<fir.box<fir.heap<fir.array<?xf32>>>>, fir.ref<fir.box<fir.heap<fir.array<?xf32>>>>            |
| CHARACTER(10) :: C                        | %mem = //  … dummy argument, or local, or global <br> %c = hlfir.declare %mem lbs %c0 {fir.def = "\_QPfooEc"} :  fir.ref<fir.char<10>>, fir.ref<fir.char<10>>                                                            |
| CHARACTER(\*) :: C                        | %unbox = fir.unbox %bochar (fir.boxchar<1>) -> (fir.ref<fir.char<?>>, index) <br> %c = hlfir.declare %unbox#0 typeparams %unbox#1 {fir.def = "\_QPfooEc"} : (index) ->  fir.boxchar<1>, fir.ref<fir.char<?>>             |
| CHARACTER(\*), OPTIONAL, ALLOCATABLE :: C | %mem = // … dummy argument <br> %c = hlfir.declare %mem {fir.def = "\_QPfooEc", fir.alloc, fir.optional, fir.assumed\_len\_alloc} :  fir.ref<fir.box<fir.heap<fir.char<?>>>>, fir.ref<fir.box<fir.heap<fir.char<?>>>>    |
| TYPE(T) :: X                              | %mem = //  … dummy argument, or local, or global <br> %x = hlfir.declare %mem {fir.def = "\_QPfooEx"} : fir.ref<fir.type<t{...}>>, fir.ref<fir.type<t{...}>>                                                             |
| TYPE(T(L)) :: X                           | %mem = //  … dummy argument, or local, or global <br> %lval = fir.load %l <br> %x = hlfir.declare %mem typeparams %lval {fir.def = "\_QPfooEx"} : fir.box<fir.type<t{...}>>, fir.box<fir.type<t{...}>>                   |
| CLASS(\*), POINTER :: X                   | %mem = //  … dummy argument, or local, or global <br> %x = hlfir.declare %mem {fir.def = "\_QPfooEx", fir.ptr} : fir.class<fir.ptr<None>>  fir.class<fir.ptr<None>>                                                      |
| REAL :: X(..)                             | %mem = //  … dummy argument <br> %x = hlfir.declare %mem {fir.def = "\_QPfooEx"} : fir.box<fir.array<..xf32>>, fir.box<fir.array<..xf32>>                                                                                |

#### fir.declare operation

Motivation: keep variable information available in FIR, at least with
the intent to be able to produce debug information.

Syntax:
```
%var = fir.declare %base [shape %extent1, %extent2, ...] [lbs %lb1, %lb2, ...] [typeparams %l1, ...] {fir.def = mangled_variable_name, attributes} : [(....) ->] T
```

%var will have the same type as %base. When no debug info is generated, the
operation can be replaced by %base when lowering to LLVM. Otherwise, the
operation is similar to hlfir.declare and will be produced from it.

#### hlfir.associate operation

Motivation: represent Fortran associations (both from variables and expressions)
and allow keeping actual/dummy argument association information after inlining.

Syntax:
```
%var = hlfir.associate %expr_or_var {fir.def = mangled_uniq_name, attributes} (AnyExprOrVarType) -> AnyVarType
```

hlfir.associate is used to represent the following associations:
- Dummy/Actual association on the caller side (the callee side uses
  hlfir.declare).
- Host association in block constructs when VOLATILE/ASYNC attributes are added
  locally
- ASSOCIATE construct (both from variable and expressions).

When the operand is a variable, hlfir.associate allows changing the attributes
of the variable locally, and to encode certain side-effects (like
copy-in/copy-out when going from a non-contiguous variable to a contiguous
variable, with the help of the related hlfir.end_association operation).

When the operand is an expression, hlfir.associate allows associating a storage
location to an expression value.

A hlfir.associate must be followed by a related hlfir.end_association that will
allow inserting any necessary finalization or copy-out later.

#### hlfir.end_association operation

Motivation: mark the place where some association should end and some side
effects might need to occur.

The hlfir.end_associate is a placeholder to later insert
deallocation/finalization if the variable was associated with an expression,
and to insert copy-out/deallocation if the variable was associated with another
variable with a copy-in.

Syntax:
```
hlfir.end_association %var [%original_variable] {attributes}
```


The attributes can be:
-   copy_out (copy out the associated variable back into the original variable
    if a copy-in occurred)
-   finalize_copy_in (deallocate the temporary storage for the associated
    variable if a copy-in occurred but the associated variable was not modified
    (e.g., it is intent(in))).
-   finalize: indicate that a finalizer should be run on the entity associated
    with the variable (There is currently no way to deduce this only from the
    variable type in FIR). It will give the finalizer mangled name so that it
    can be later called.

If the copy_out or finalize_copy_in attribute is set, “original_variable” (the
argument of the hlfir.associate that produced %var) must be provided. The
rationale is that the original variable address is needed to verify if a
temporary was created, and if needed, to copy the data back to it.

#### hlfir.finalize

Motivation: mark end of life of local variables

Mark the place where a local variable will go out of scope. The main goal is to
retain this information even after local variables are inlined.

Syntax:
```
hlfir.finalize %var {attributes}
```

The attributes can be:
-   finalize: indicate that a finalizer should be run on the entity associated
    with the variable (There is currently no way to deduce this only from the
    variable type in FIR).

Note that finalization will not free the local variable storage if it was
allocated on the heap. If lowering created the storage passed to hlfir.declare
via a fir.allocmem, lowering should insert a fir.freemem after the
hlfir.finalize.  This could help making fir.allocmem to fir.alloca promotion
simpler, and also because finalization may be run without the intent to
deallocate the variable storage (like on INTENT(OUT) dummies).


#### hlfir.designate

Motivation: Represent designators at a high-level and allow representing some
information about derived type components that would otherwise be lost, like
component lower bounds.

Represent Fortran designators in a verbatim way: both triplet, and component
parts.

Syntax:
```
%var = hlfir.designate %base [“component”,] [(%i, %k:l%:%m)] [substr ub, lb] [imag|real] [shape extent1, extent2, ....] [lbs lb1, lb2, .....] [typeparams %l1, ...] {attributes}
```

hlfir.designate is intended to encode a single part-ref (as defined by the
fortran standard). That means that a(:)%x(i, j, k) must be split into two
hlfir.designate: one for a(:), and one for x(i, j, k).  If the base is ranked,
and the component is an array, the subscripts are mandatory and must not
contain triplets. This ensures that the result of a fir.designator cannot be a
"super-array".

The subscripts passed to hlfir.designate must be based on the base lower bounds
(one by default).

A substring is built by providing the lower and upper character indices after
`substr`. Implicit substring bounds must be made explicit by lowering.  It is
not possible to provide substr if a component is already provided. Instead the
related Fortran designator must be split into two fir.designator. This is
because the component character length will be needed to compute the right
stride, and it might be lost if not placed on the first designator typeparams.

Real and Imaginary complex parts are represented by an optional imag or real
tag. It can be added even if there is already a component.

The shape, lower bound, and type parameter operands represent the output entity
properties. The point of having those made explicit is to allow early folding
and hoisting of array section shape and length parameters (which especially in
FORALL contexts, can simplify later assignment temporary insertion a lot). Also,
if lower bounds of a derived type component array could not be added here, they
would be lost since they are not represented by other means in FIR (the fir.type
does not include this information).

hlfir.designate is not intended to describe vector subscripted variables.
Instead, lowering will have to introduce loops to do element by element
addressing. See the Examples section. This helps keeping hlfir.designate simple,
and since the contexts where a vector subscripted entity is considered to be a
variable (in the sense that it can be modified) are very limited, it seems
reasonable to have lowering deal with this aspect. For instance, a vector
subscripted entity cannot be passed as a variable, it cannot be a pointer
assignment target, and when it appears as an associated entity in an ASSOCIATE,
the related variable cannot be modified.

#### hlfir.assign

Motivation: represent assignment at a high-level (mainly a change for array and
character assignment) so that optimization pass can clearly reason about it
(value propagation, inserting temporary for right-hand side evaluation only when
needed), and that lowering does not have to implement it all.

Syntax:
```
hlfir.assign %expr_or_var to %var [attributes]
```

The attributes can be:

-   realloc: mark that assignment has F2003 semantics and that the left-hand
    side may have to be deallocated/reallocated…
-   use_assign=@function: mark a user defined assignment
-   no_overlap: mark that an assignment does not need a temporary (added by an
    analysis pass).
-   unordered : mark that an assignment can happen in any element order (not
    true if there is an impure elemental function being called).
-   temporary_lhs: mark that the left hand side of the assignment is
    a compiler generated temporary.

This will replace the current array_load/array_access/array_merge semantics.
Instead, a more generic alias analysis will be performed on the LHS and RHS to
detect aliasing, and a temporary inserted if needed. The alias analysis will
look at all the memory references in the RHS operand tree and base overlap
decisions on the related variable declaration operations. This same analysis
should later allow moving/merging some expression evaluation between different
statements.

Note about user defined assignments: semantics is resolving them and building
the related subroutine call. So a fir.call could directly be made in lowering if
the right hand side was always evaluated in a temporary. The motivation to use
hlfir.assign is to help the temporary removal, and also to deal with two edge
cases: user assignment in a FORALL (the forall pass will need to understand that
this an assignment), and allocatable assignment mixed with user assignment
(implementing this as a call in lowering would require lowering the whole
reallocation logic in lowering already, duplicating the fact that hlfir.assign
should deal with it).

#### hlfir.ptr_assign

Motivation: represent pointer assignment without lowering the exact pointer
implementation (descriptor address, fir.ref<fir.box> or simple pointer scalar
fir.llvm_ptr<fir.ptr>).

Syntax:
```
hlfir.ptr_assign %var [[reshape %reshape] | [lbounds %lb1, …., %lbn]] to %ptr
```

It is important to keep pointer assignment at a high-level so that they can
later correctly be processed in hlfir.forall.

#### hlfir.allocate

Motivation: keep POINTER and ALLOCATABLE allocation explicit in HLFIR, while
allowing later lowering to either inlined fir.allocmem or Fortran runtime
calls. Generating runtime calls allow the runtime to do Fortran specific
bookkeeping or flagging and to provide better runtime error reports.

The main difference with the ALLOCATE statement is that one distinct
hlfir.allocate has to be created for each element of the allocation-list.
Otherwise, it is a naive lowering of the ALLOCATE statement.

Syntax:
```
%stat = hlfir.allocate %var [%shape] [%type_params] [[src=%source] | [mold=%mold]] [errmsg =%errmsg]
```

#### hlfir.deallocate

Motivation: keep deallocation explicit in HLFIR, while allowing later lowering
to Fortran runtime calls to allow the runtime to do Fortran specific
bookkeeping or flagging of allocations.

Similarly to hlfir.allocate, one operation must be created for each
allocate-object-list object.

Syntax:
```
%stat = hlfir.deallocate %var [errmsg=err].
```

####  hlfir.elemental

Motivation: represent elemental operations without defining array level
operations for each of them, and allow the representation of array expressions
as function of the indices.

The hlfir.elemental operation can be seen as a closure: it is defining a
function of the indices that returns the value of the element of the
represented array expression at the given indices. This an operation with an
MLIR region. It allows detailing how an elemental expression is implemented at
the element level, without yet requiring materializing the operands and result
in memory.  The hlfir.expr<T> elements value can be obtained using hlfir.apply.

The element result is built with a fir.result op, whose result type can be a
scalar hlfir.expr<T> or any scalar constant size types (e.g. i32, or f32).

Syntax:
```
%op = hlfir.elemental (%indices) %shape [%type_params] [%dynamic_type] {
  ….
  fir.result %result_element
}
```


Note that %indices are not operands, they are the elemental region block
arguments, representing the array iteration space in a one based fashion.
The choice of using one based indicies is to match Fortran default for
array variables, so that there is no need to generate bound adjustments
when working with one based array variables in an expression.

Illustration: “A + B” represented with a hlfir.elemental.

```
%add = hlfir.elemental (%i:index, %j:index) shape %shape (!fir.shape<2>) -> !hlfir.expr<?x?xf32> {
  %belt = hlfir.designate %b, %i, %j : (!fir.box<!fir.array<?x?xf32>>, index, index) -> !fir.ref<f32>
  %celt = hlfir.designate %c, %i, %j : (!fir.box<!fir.array<?x?xf32>>, index, index) -> !fir.ref<f32>
  %bval = fir.load %belt : (!fir.ref<f32>) -> f32
  %cval = fir.load %celt : (!fir.ref<f32>) -> f32
  %add = arith.addf %bval, %cval : f32
  fir.result %res : f32
}
```

In contexts where it can be proved that the array operands were not modified
between the hlfir.elemental and the hlfir.apply, the region of the
hlfir.elemental can be inlined at the hlfir.apply. Otherwise, if there is no
such guarantee, or if the hlfir.elemental is not “visible” (because its result
is passed as a block argument), the hlfir.elemental will be lowered to an array
temporary. This will be done as a HLFIR to HLFIR optimization pass. Note that
MLIR inlining could be used if hlfir.elemental implemented the
CallableInterface and hlfir.apply the CallInterface.  But MLIR generic inlining
is probably too generic for this case: no recursion is possible here, the call
graphs are trivial, and using MLIR inlining here could introduce later
conflicts or make normal function inlining more complex because FIR inlining
hooks would already be used.

hlfir.elemental allows delaying elemental array expression buffering and
combination. Its generic aspect has two advantages:
- It avoids defining one operation per elemental operation or intrinsic,
  instead, the related arith dialect operations can be used directly in the
  elemental regions. This avoids growing HLFIR and having to maintain about a
  hundred operations.
- It allows representing transformational intrinsics as functions of the indices
  while doing optimization as described in
  [Array Composition](ArrayComposition.md). This because the indices can be
  transformed inside the region before being applied to array variables
  according to any kind of transformation (semi-affine or not).


#### Introducing the hlfir.apply operation

Motivation: provide a way to get the element of an array expression
(hlfir.expr<?x…xT>)

This is the addressing equivalent for expressions. A notable difference is that
it can only take simple scalar indices (no triplets) because it is not clear
why supporting triplets would be needed, and keeping the indexing simple makes
inlining of hlfir.elemental much easier.

If hlfir.elemental inlining is not performed, or if the hlfir.expr<T> array
expression is produced by another operation (like fir.intrinsic) that is not
rewritten, hlfir.apply will be lowered to an actual addressing operation that
will address the temporary that was created for the hlfir.expr<T> value that
was materialized in memory.

hlfir.apply indices will be one based to make further lowering simpler.

Syntax:
```
%element = hlfir.apply %array_expr %i, %j: (hlfir.expr<?x?xi32>) -> i32
```

#### Introducing operations for transformational intrinsic functions

Motivation: Represent transformational intrinsics functions at a high-level so
that they can be manipulated easily by the optimizer, and do not require
materializing the result as a temporary in lowering.

An operation will be added for each Fortran transformational functions (SUM,
MATMUL, TRANSPOSE....). It translates the Fortran expression verbatim: it takes
the same number of arguments as the Fortran intrinsics and returns a
hlfir.expr<T>. The arguments may be hlfir.expr<T>, simple scalar types (e.g.,
i32, f32), or variables.

The exception being that the arguments that are statically absent would be
passed to it (passing results of fir.absent operation), so that the arguments
can be identified via their positions.

This operation is meant for the transformational intrinsics, not the elemental
intrinsics, that will be implemented using hlfir.elemental + mlir math dialect
operations, nor the intrinsic subroutines (like random_seed or system_clock),
that will be directly lowered in lowering.

Syntax:
```
%res = hlfir."intrinsic_name" %expr_or_var, ...
```

These operations will all inherit a same operation base in tablegen to make
their definition and identification easy.

Without any optimization, codegen would then translate the operations to
exactly the same FIR as currently generated by IntrinsicCall.cpp (runtime calls
or inlined code with temporary allocation for array results). The fact that
they are the verbatim Fortran translations should allow to move the lowering
code to a translation pass without massive changes.

An operation will at least be created for each of the following transformational
intrinsics: all, any, count, cshift, dot_product, eoshift, findloc, iall, iany,
iparity, matmul, maxloc, maxval, minloc, minval, norm2, pack, parity, product,
reduce, repeat, reshape, spread, sum, transfer, transpose, trim, unpack.

For the following transformational intrinsics, the current lowering to runtime
call will probably be used since there is little point to keep them high level:
- command_argument_count, get_team, null, num_images, team_number, this_image
  that are more program related (and cannot appear for instance in constant
  expressions)
- selected_char_kind, selected_int_kind, selected_real_kind that returns scalar
  integers

#### Introducing operations for composed intrinsic functions

Motivation: optimize commonly composed intrinsic functions (e.g.
MATMUL(TRANSPOSE(a), b)). This optimization is implemented in Classic Flang.

An operation and runtime function will be added for each commonly used
composition of intrinsic functions. The operation will be the canonical way to
write this chained operation (the MLIR canonicalization pass will rewrite the
operations for the composed intrinsics into this one operation).

These new operations will be treated as though they were standard
transformational intrinsic functions.

The composed intrinsic operation will return a hlfir.expr<T>. The arguments
may be hlfir.expr<T>, boxed arrays, simple scalar types (e.g. i32, f32), or
variables.

To keep things simple, these operations will only match one form of the composed
intrinsic functions: therefore there will be no optional arguments.

Syntax:
```
%res = hlfir."intrinsic_name" %expr_or_var, ...
```

The composed intrinsic operation will be lowered to a `fir.call` to the newly
added runtime implementation of the operation.

These operations should not be added where the only improvement is to avoid
creating a temporary intermediate buffer which would otherwise be removed by
intelligent bufferization of a hlfir.expr. Similarly, these should not replace
profitable uses of hlfir.elemental.

#### Introducing operations for character operations and elemental intrinsic functions


Motivation: represent character operations without requiring the operand and
results to be materialized in memory.

fir.char_op is intended to represent:
-  Character concatenation (//)
-  Character MIN/MAX
-  Character MERGE
-  “SET_LENGTH”
-  Character conversions
-  REPEAT
-  INDEX
-  CHAR
-  Character comparisons
-  LEN_TRIM

The arguments must be scalars, the elemental aspect should be handled by a
hlfir.elemental operation.

Syntax:
```
%res = hlfir.“char_op” %expr_or_var
```

Just like for the transformational intrinsics, if no optimization occurs, these
operations will be lowered to memory operations with temporary results (if the
result is a character), using the same generation code as the one currently used
in lowering.

#### hlfir.array_ctor

Motivation: represent array constructor without creating temporary

Many array constructors have a limited number of elements (less than 10), the
current lowering of array constructor is rather complex because it must deal
with the generic cases.

Having a representation to represent array constructor will allow an easier
lowering of array constructor, and make array ctor a lot easier to manipulate.
For instance, for small array constructors, loops could could be unrolled with
the array ctor elements without ever creating a dynamically allocated array
temporary and loop nest using it.

Syntax:
```
%array_ctor = hlfir.array_ctor %expr1, %expr2 ….
```

Note that hlfir.elemental could be used to implement some ac-implied-do,
although this is not yet clarified since ac-implied-do may contain more than
one scalar element (they may contain a list of scalar and array values, which
would render the representation in a hlfir.elemental tricky, but maybe not
impossible using if/then/else and hlfir.elemental nests using the index value).
One big issue though is that hlfir.elemental requires the result shape to be
pre-computed (it is an operand), and with an ac-implied-do containing user
transformational calls returning allocatable or pointer arrays, it is
impossible to pre-evaluate the shape without evaluating all the function calls
entirely (and therefore all the array constructor elements).

#### hlfir.get_extent

Motivation: inquire about the extent of a hlfir.expr, variable, or fir.shape

Syntax:
```
%extent = hlfir.get_extent %shape_expr_or_var, dim
```

dim is a constant integer attribute.

This allows inquiring about the extents of expressions whose shape may not be
yet computable without generating detailed, low level operations (e.g, for some
transformational intrinsics), or to avoid going into low level details for
pointer and allocatable variables (where the descriptor needs to be read and
loaded).

#### hlfir.get_typeparam

Motivation: inquire about the type parameters of a hlfir.expr, or variable.

Syntax:
```
%param = hlfir.get_typeparam %expr_or_var [, param_name]
```
- param_name is an optional string attribute that must contain the length
  parameter name if %expr_or_var is a derived type.

####  hlfir.get_dynamic_type

Motivation: inquire about the dynamic type of a polymorphic hlfir.expr or
variable.

Syntax:
```
%dynamic_type = hlfir.get_dynamic_type %expr_or_var
```

#### hlfir.get_lbound

Motivation: inquire about the lower bounds of variables without digging into
the implementation details of pointers and allocatables.

Syntax:
```
%lb = hlfir.get_lbound %var, n
```

Note: n is an integer constant attribute for the (zero based) dimension.

####  hlfir.shape_meet

Motivation: represent conformity requirement/information between two array
operands so that later optimization can choose the best shape information
source, or insert conformity runtime checks.

Syntax:
```
%shape = hlfir.shape_meet %shape1, %shape2
```

Suppose A(n), B(m) are two explicit shape arrays. Currently, when A+B is
lowered, lowering chose which operand shape gives the result shape information,
and it is later not retrievable that both n and m can be used. If lowering
chose n, but m later gets folded thanks to inlining or constant propagation, the
optimization passes have no way to use this constant information to optimize the
result storage allocation or vectorization of A+B.  hlfir.shape_meet intends to
delay this choice until constant propagation or inlining can provide better
information about n and m.

#### hlfir.forall

Motivation: segregate the Forall lowering complexity in its own unit.

Forall is tough to lower because:
-   Lowering it in an optimal way requires analyzing several assignments/mask
    expressions.
-   The shape of the temporary needed to store intermediate evaluation values is
    not a Fortran array in the general case, and cannot in the general case be
    maximized/pre-computed without executing the forall to compute the bounds of
    inner forall, and the shape of the assignment operands that may depend on
    the bound values.
-   Mask expressions evaluation should be affected by previous assignment
    statements, but not by the following ones. Array temporaries may be
    required for the masks to cover this.
-   On top of the above points, Forall can contain user assignments, pointer
    assignments, and assignment to whole allocatable.


The hlfir.forall syntax would be exactly the one of a fir.do_loop. The
difference would be that hlfir.assign and hlfir.ptr_assign inside hlfir.forall
have specific semantics (the same as in Fortran):
-   Given one hlfir.assign, all the iteration values of the LHS/RHS must be
    evaluated before the assignment of any value is done.
-   Given two hlfir.assign, the first hlfir.assign must be fully performed
    before any evaluation of the operands of the second assignment is done.
-   Masks (fir.if arguments), if any, should be evaluated before any nested
    assignments. Any assignments syntactically before the where mask occurrence
    must be performed before the mask evaluation.

Note that forall forbids impure function calls, hence, no calls should modify
any other expression evaluation and can be removed if unused.

The translation of hlfir.forall will happen by:
-   1. Determining if the where masks value may be modified by any assignments
    - Yes, pre-compute all masks in a pre-run of the forall loop, creating
      a “forall temps” (we may need a FIR concept to help here).
    - No, Do nothing (or indicate it is safe to evaluate masks while evaluating
      the rest).
-   2. Determining if a hlfir.assign operand expression depends on the
       previous hlfir.assign left-hand side base value.
    - Yes, split the hlfir.assign into their own nest of hlfir.forall loops.
    - No, do nothing (or indicate it is safe to evaluate the assignment while
      evaluating previous assignments)
-   3. For each assignments, check if the RHS/LHS operands value may depend
     on the LHS base:
    - Yes, split the forall loops. Insert a “forall temps” before the loops for
      the “smallest” part that may overlap (which may be the whole RHS, or some
      RHS sub-part, or some LHS indices). In the first nest, evaluate this
      overlapping part into the temp. In the next forall loop nest, modify the
      assignment to use the temporary, and add the [no_overlap] flag to indicate
      no further temporary is needed. Insert code to finalize the temp after its
      usage.

## New HLFIR Transformation Passes

### Mandatory Passes (translation towards lower-level representation)

Note that these passes could be implemented as a single MLIR pass, or successive
passes.

-   Forall rewrites (getting rid of hlfir.forall)
-   Array assignment rewrites (getting rid of array hlfir.assign)
-   Bufferization: expression temporary materialization (getting rid of
    hlfir.expr, and all the operations that may produce it like transformational
    intrinsics and hlfir.elemental, hlfir.apply).
-   Call interface argument association lowering (getting rid of hlfir.associate
    and hlfir.end_associate)
-   Lowering high level operations using variables into FIR operations
    operating on memory (translating hlfir.designate, scalar hlfir.assign,
    hlfir.finalize into fir.array_coor, fir.do_loop, fir.store, fir.load.
    fir.embox/fir.rebox operations).

Note that these passes do not have to be the first one run after lowering. It is
intended that CSE, DCE, algebraic simplification, inlining and some other new
high-level optimization passes discused below be run before doing any of these
translations.

After that, the current FIR pipeline could be used to continue lowering towards
LLVM.

### Optimization Passes

-   Elemental expression inlining (inlining of hlfir.elemental in hlfir.apply)
-   User function Inlining
-   Transformational intrinsic rewrites as hlfir.elemental expressions
-   Assignments propagation
-   Shape/Rank/dynamic type propagation

These high level optimization passes can be run any number of times in any
order.

## Transition Plan

The new higher-level steps proposed in this document will require significant
refactoring of lowering. Codegen should not be impacted since the current FIR
will remain untouched.

A lot of the code in lowering generating Fortran features (like an intrinsic or
how to do assignments) is based on the fir::ExtendedValue concept. This
currently is a collection of mlir::Value that allows describing a Fortran object
(either a variable or an evaluated expression result). The variable and
expression concepts described above should allow to keep an interface very
similar to the fir::ExtendedValue, but having the fir::ExtendedValue wrap a
single value or mlir::Operation* from which all of the object entity
information can be inferred.

That way, all the helpers currently generating FIR from fir::ExtendedValue could
be kept and used with the new variable and expression concepts with as little
modification as possible.

The proposed plan is to:
- 1. Introduce the new HLFIR operations.
- 2. Refactor fir::ExtendedValue so that it can work with the new variable and
     expression concepts (requires part of 1.).
- 3. Introduce the new translation passes, using the fir::ExtendedValue helpers
     (requires 1.).
- 3.b Introduce the new optimization passes (requires 1.).
- 4. Introduce the fir.declare and hlfir.finalize usage in lowering (requires 1.
     and 2. and part of 3.).

The following steps might have to be done in parallel of the current lowering,
to avoid disturbing the work on performance until the new lowering is complete
and on par.

- 5. Introduce hlfir.designate and hlfir.associate usage in lowering.
- 6. Introduce lowering to hlfir.assign (with RHS that is not a hlfir.expr),
     hlfir.ptr_assign.
- 7. Introduce lowering to hlfir.expr and related operations.
- 8. Introduce lowering to hlfir.forall.

At that point, lowering using the high-level FIR should be in place, allowing
extensive testing.
- 9. Debugging correctness.
- 10. Debugging execution performance.

The plan is to do these steps incrementally upstream, but for lowering this will
most likely be safer to do have the new expression lowering implemented in
parallel upstream, and to add an option to use the new lowering rather than to
directly modify the current expression lowering and have it step by step
equivalent functionally and performance wise.

## Examples

### Example 1: simple array assignment

```Fortran
subroutine foo(a, b)
  real :: a(:), b(:)
  a = b
end subroutine
```

Lowering output:

```
func.func @_QPfoo(%arg0: !fir.box<!fir.array<?xf32>>, %arg1: !fir.box<!fir.array<?xf32>>) {
  %a = hlfir.declare %arg0 {fir.def = "_QPfooEa"} : !fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>
  %b = hlfir.declare %arg1 {fir.def = "_QPfooEb"} : !fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>
  hlfir.assign %b#0 to %a#0 : !fir.box<!fir.array<?xf32>>
  return
}
```

HLFIR array assignment lowering pass:
-   Query: can %b value depend on %a? No, they are two different argument
    associated variables that are neither target nor pointers.
-   Lower to assignment to loop:

```HFLIR
func.func @_QPfoo(%arg0: !fir.box<!fir.array<?xf32>>, %arg1: !fir.box<!fir.array<?xf32>>) {
  %a = hlfir.declare %arg0 {fir.def = "_QPfooEa"} : !fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>
  %b = hlfir.declare %arg1 {fir.def = "_QPfooEb"} : !fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>

  %ashape = hlfir.shape_of %a#0
  %bshape = hlfir.shape_of %b#0
  %shape = hlfir.shape_meet %ashape, %bshape
  %extent = hlfir.get_extent %shape, 0

  %c1 = arith.constant 1 : index

  fir.do_loop %i = %c1 to %extent step %c1 unordered {
    %belt = hlfir.designate %b#0, %i
    %aelt = hlfir.designate %a#0, %i
    hlfir.assign %belt to %aelt : fir.ref<f32>, fir.ref<f32>
  }
  return
}
```

HLFIR variable operations to memory translation pass:
-   hlfir.designate is rewritten into fir.array_coor operation on the variable
    associated memory buffer, and returns the element address
-   For numerical scalar, hlfir.assign is rewritten to fir.store (and fir.load
    of the operand if needed), for derived type and characters, memory copy
    (and padding for characters) is done.
-   hlfir.shape_of are lowered to fir.box_dims, here, no constant information
    was obtained from any of the source shape, so hlfir.shape_meet is a no-op,
    selecting the first shape (a conformity runtime check could be inserted
    under debug options).
-   hlfir.declare are translated into fir.declare that are no-ops and will allow
    generating debug information for LLVM.

This pass would wrap operations defining variables (hlfir.declare/hlfir.designate)
as fir::ExtendedValue, and use all the current helpers operating on it
(e.g.: fir::factory::genScalarAssignment).

```
func.func @_QPfoo(%arg0: !fir.box<!fir.array<?xf32>>, %arg1:
  !fir.box<!fir.array<?xf32>>) {
  %a = fir.declare %arg0 {fir.def = "_QPfooEa"} : !fir.box<!fir.array<?xf32>>
  %b = fir.declare %arg1 {fir.def = "_QPfooEb"} : !fir.box<!fir.array<?xf32>>
  %c1 = arith.constant 1 : index
  %dims = fir.box_dims %a, 1
  fir.do_loop %i = %c1 to %dims#1 step %c1 unordered {
    %belt = fir.array_coor %b, %i : (!fir.box<!fir.array<?xf32>>, index) -> fir.ref<f32>
    %aelt = fir.array_coor %a, %i : (!fir.box<!fir.array<?xf32>>, index) -> fir.ref<f32>
    %bval = fir.load %belt : f32
    fir.store %bval to %aelt : fir.ref<f32>
  }
  return
}
```

This reaches the current FIR level (except fir.declare that can be kept until
LLVM codegen and dropped on the floor if there is no debug information
generated).

### Example 2: array assignment with elemental expression

```Fortran
subroutine foo(a, b, p, c)
  real, target :: a(:)
  real :: b(:), c(100)
  real, pointer :: p(:)
  a = b*p + c
end subroutine
```

Lowering output:

```
func.func @_QPfoo(%arg0: !fir.box<!fir.array<?xf32>>, %arg1: !fir.box<!fir.array<?xf32>>, %arg2: !fir.box<!fir.ptr<!fir.array<?xf32>>>, %arg3: !fir.ref<!fir.array<100xf32>>) {
  %a = hlfir.declare %arg0 {fir.def = "_QPfooEa"} {fir.target} : !fir.box<!fir.array<?xf32>, !fir.box<!fir.array<?xf32>
  %b =  hlfir.declare %arg1 {fir.def = "_QPfooEb"} : !fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>
  %p = hlfir.declare %arg2 {fir.def = "_QPfooEp", fir.ptr} : !fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.box<!fir.ptr<!fir.array<?xf32>>>
  %c =  hlfir.declare %arg3 {fir.def = "_QPfooEc"} : !fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>
  %bshape = hlfir.shape_of %b#0
  %pshape = hlfir.shape_of %p#0
  %shape1 = hlfir.shape_meet %bshape, %pshape
  %mul = hlfir.elemental(%i:index) %shape1 {
    %belt = hlfir.designate %b#0, %i
    %p_lb = hlfir.get_lbound %p#0, 1
    %i_zero = arith.subi %i, %c1
    %i_p = arith.addi %i_zero,  %p_lb
    %pelt = hlfir.designate %p#0, %i_p
    %bval = fir.load %belt : f32
    %pval = fir.load %pelt : f32
    %mulres = arith.mulf %bval, %pval : f32
     fir.result %mulres : f32
  }
  %cshape = hlfir.shape_of %c
  %shape2 = hlfir.shape_meet %cshape, %shape1
  %add =  hlfir.elemental(%i:index) %shape2 {
    %mulval = hlfir.apply %mul, %i : f32
    %celt = hlfir.designate %c#0, %i
    %cval = fir.load %celt
    %add_res = arith.addf %mulval, %cval
    fir.result %add_res
  }
  hlfir.assign %add to %a#0 : hlfir.expr<?xf32>, !fir.box<!fir.array<?xf32>
  return
}
```

Step 1: hlfir.elemental inlining: inline the first hlfir.elemental into the
second one at the hlfir.apply.


```
func.func @_QPfoo(%arg0: !fir.box<!fir.array<?xf32>>, %arg1: !fir.box<!fir.array<?xf32>>, %arg2: !fir.box<!fir.ptr<!fir.array<?xf32>>>, %arg3: !fir.ref<!fir.array<100xf32>>) {
  %a = hlfir.declare %arg0 {fir.def = "_QPfooEa"} {fir.target} : !fir.box<!fir.array<?xf32>, !fir.box<!fir.array<?xf32>
  %b =  hlfir.declare %arg1 {fir.def = "_QPfooEb"} : !fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>
  %p = hlfir.declare %arg2 {fir.def = "_QPfooEp", fir.ptr} : !fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.box<!fir.ptr<!fir.array<?xf32>>>
  %c =  hlfir.declare %arg3 {fir.def = "_QPfooEc"} : !fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>
  %bshape = hlfir.shape_of %b#0
  %pshape = hlfir.shape_of %p#0
  %shape1 = hlfir.shape_meet %bshape, %pshape
  %cshape = hlfir.shape_of %c
  %shape2 = hlfir.shape_meet %cshape, %shape1
  %add =  hlfir.elemental(%i:index) %shape2 {
    %belt = hlfir.designate %b#0, %i
    %p_lb = hlfir.get_lbound %p#0, 1
    %i_zero = arith.subi %i, %c1
    %i_p = arith.addi %i_zero,  %p_lb
    %pelt = hlfir.designate %p#0, %i_p
    %bval = fir.load %belt : f32
    %pval = fir.load %pelt : f32
    %mulval = arith.mulf %bval, %pval : f32
    %celt = hlfir.designate %c#0, %i
    %cval = fir.load %celt
    %add_res = arith.addf %mulval, %cval
    fir.result %add_res
  }
  hlfir.assign %add to %a#0 : hlfir.expr<?xf32>, !fir.box<!fir.array<?xf32>
  return
}
```

Step2: alias analysis around the array assignment:

-   May %add value depend on %a variable?
-   Gather variable and function calls in %add operand tree (visiting
    hlfir.elemental regions)
-   Gather references to %b, %p, and %c. %p is a pointer variable according to
    its defining operations. It may alias with %a that is a target. -> answer
    yes.
-   Insert temporary, and duplicate array assignments, that can be lowered to
    loops at that point

Note that the alias analysis could have already occurred without inlining the
%add hlfir.elemental.


```
func.func @_QPfoo(%arg0: !fir.box<!fir.array<?xf32>>, %arg1: !fir.box<!fir.array<?xf32>>, %arg2: !fir.box<!fir.ptr<!fir.array<?xf32>>>, %arg3: !fir.ref<!fir.array<100xf32>>) {
  %a = hlfir.declare %arg0 {fir.def = "_QPfooEa"} {fir.target} : !fir.box<!fir.array<?xf32>, !fir.box<!fir.array<?xf32>
  %b =  hlfir.declare %arg1 {fir.def = "_QPfooEb"} : !fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>
  %p = hlfir.declare %arg2 {fir.def = "_QPfooEp", fir.ptr} : !fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.box<!fir.ptr<!fir.array<?xf32>>>
  %c =  hlfir.declare %arg3 {fir.def = "_QPfooEc"} : !fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>
  %bshape = hlfir.shape_of %b#0
  %pshape = hlfir.shape_of %p#0
  %shape1 = hlfir.shape_meet %bshape, %pshape
  %cshape = hlfir.shape_of %c
  %shape2 = hlfir.shape_meet %cshape, %shape1
  %add =  hlfir.elemental(%i:index) %shape2 {
    %belt = hlfir.designate %b#0, %i
    %p_lb = hlfir.get_lbound %p#0, 1
    %i_zero = arith.subi %i, %c1
    %i_p = arith.addi %i_zero, %p_lb
    %pelt = hlfir.designate %p#0, %i_p
    %bval = fir.load %belt : f32
    %pval = fir.load %pelt : f32
    %mulval = arith.mulf %bval, %pval : f32
    %celt = hlfir.designate %c#0, %i
    %cval = fir.load %celt
    %add_res = arith.addf %mulval, %cval
    fir.result %add_res
  }
  %extent = hlfir.get_extent %shape2, 0: (fir.shape<1>) -> index
  %tempstorage = fir.allocmem %extent : fir.heap<fir.array<?xf32>>
  %temp = hlfir.declare %tempstorage, shape %extent {fir.def = QPfoo.temp001} : (index) -> fir.box<fir.array<?xf32>>, fir.heap<fir.array<?xf32>>
  hlfir.assign %add to %temp#0 no_overlap : hlfir.expr<?xf32>, !fir.box<!fir.array<?xf32>>
  hlfir.assign %temp to %a#0 : no_overlap  : !fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>
  hlfir.finalize %temp#0
  fir.freemem %tempstorage
  return
}
```

Step 4: Lower assignments to regular loops since they have the no_overlap
attribute, and inline the hlfir.elemental into the first loop nest.

```
func.func @_QPfoo(%arg0: !fir.box<!fir.array<?xf32>>, %arg1: !fir.box<!fir.array<?xf32>>, %arg2: !fir.box<!fir.ptr<!fir.array<?xf32>>>, %arg3: !fir.ref<!fir.array<100xf32>>) {
  %a = hlfir.declare %arg0 {fir.def = "_QPfooEa"} {fir.target} : !fir.box<!fir.array<?xf32>, !fir.box<!fir.array<?xf32>
  %b =  hlfir.declare %arg1 {fir.def = "_QPfooEb"} : !fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>
  %p = hlfir.declare %arg2 {fir.def = "_QPfooEp", fir.ptr} : !fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.box<!fir.ptr<!fir.array<?xf32>>>
  %c =  hlfir.declare %arg3 {fir.def = "_QPfooEc"} : !fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>
  %bshape = hlfir.shape_of %b#0
  %pshape = hlfir.shape_of %p#0
  %shape1 = hlfir.shape_meet %bshape, %pshape
  %cshape = hlfir.shape_of %c
  %shape2 = hlfir.shape_meet %cshape, %shape1
  %extent = hlfir.get_extent %shape2, 0: (fir.shape<1>) -> index
  %tempstorage = fir.allocmem %extent : fir.heap<fir.array<?xf32>>
  %temp = hlfir.declare %tempstorage, shape %extent {fir.def = QPfoo.temp001} : (index) -> fir.box<fir.array<?xf32>>, fir.heap<fir.array<?xf32>>
  fir.do_loop %i = %c1 to %shape2 step %c1 unordered {
    %belt = hlfir.designate %b#0, %i
    %p_lb = hlfir.get_lbound %p#0, 1
    %i_zero = arith.subi %i, %c1
    %i_p = arith.addi %i_zero,  %p_lb
    %pelt = hlfir.designate %p#0, %i_p
    %bval = fir.load %belt : f32
    %pval = fir.load %pelt : f32
    %mulval = arith.mulf %bval, %pval : f32
    %celt = hlfir.designate %c#0, %i
    %cval = fir.load %celt
    %add_res = arith.addf %mulval, %cval
    %tempelt = hlfir.designate %temp#0, %i
    hlfir.assign %add_res to %tempelt : f32, fir.ref<f32>
  }
  fir.do_loop %i = %c1 to %shape2 step %c1 unordered {
    %aelt = hlfir.designate %a#0, %i
    %tempelt = hlfir.designate %temp#0, %i
    hlfir.assign %add_res to %tempelt : f32, fir.ref<f32>
  }
  hlfir.finalize %temp#0
  fir.freemem %tempstorage
  return
}
```

Step 5 (may also occur earlier or several times): shape propagation.
-   %shape2 can be inferred from %cshape that has constant shape: the
    hlfir.shape_meet results can be replaced by it, and if the option is set,
    conformance checks can be added for %a, %b and %p.
-   %temp is small, and its fir.allocmem can be promoted to a stack allocation

```
func.func @_QPfoo(%arg0: !fir.box<!fir.array<?xf32>>, %arg1: !fir.box<!fir.array<?xf32>>, %arg2: !fir.box<!fir.ptr<!fir.array<?xf32>>>, %arg3: !fir.ref<!fir.array<100xf32>>) {
  // .....
  %cshape = fir.shape %c100
  %extent = %c100
  // updated fir.alloca
  %tempstorage = fir.alloca %extent : fir.ref<fir.array<100xf32>>
  %temp = hlfir.declare %tempstorage, shape %extent {fir.def = QPfoo.temp001} : (index) -> fir.box<fir.array<?xf32>>, fir.heap<fir.array<?xf32>>
  fir.do_loop %i = %c1 to %c100 step %c1 unordered {
    // ...
  }
  fir.do_loop %i = %c1 to %c100 step %c1 unordered {
    // ...
  }
  hlfir.finalize %temp#0
  // deleted fir.freemem %tempstorage
  return
}
```

Step 6: lower hlfir.designate/hlfir.assign in a translation pass:

At this point, the representation is similar to the current representation after
the array value copy pass, and the existing FIR flow is used (lowering
fir.do_loop to cfg and doing codegen to LLVM).

### Example 3: assignments with vector subscript

```Fortran
subroutine foo(a, b, v)
  real :: a(*), b(*)
  integer :: v(:)
  a(v) = b(v)
end subroutine
```

Lowering of vector subscripted entities would happen as follow:
- vector subscripted entities would be lowered as a hlfir.elemental implementing
  the vector subscript addressing.
- If the vector appears in a context where it can be modified (which can only
  be an assignment LHS, or in input IO), lowering could transform the
  hlfir.elemental into hlfir.forall (for assignments), or a fir.iter_while (for
  input IO) by inlining the elemental body into the created loops, and
  identifying the hlfir.designate producing the result.

```HFLFIR
func.func @_QPfoo(%arg0: !fir.ref<!fir.array<?xf32>>, %arg1: !fir.ref<!fir.array<?xf32>>, %arg2: !fir.box<<!fir.array<?xi32>>) {
  %a = hlfir.declare %arg0 {fir.def = "_QPfooEa"} : !fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>
  %b = hlfir.declare %arg1 {fir.def = "_QPfooEb"} : !fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>
  %v = hlfir.declare %arg2 {fir.def = "_QPfooEv"} : !fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>
  %vshape = hlfir.shape_of %v : fir.shape<1>
  %bsection =  hlfir.elemental(%i:index) %vshape : (fir.shape<1>) -> hlfir.expr<?xf32> {
    %v_elt = hlfir.designate %v#0, %i : (!fir.box<!fir.array<?xi32>>, index) -> fir.ref<i32>
    %v_val = fir.load %v_elt : fir.ref<i32>
    %cast = fir.convert %v_val : (i32) -> index
    %b_elt = hlfir.designate %b#0, %v_val : (!fir.ref<!fir.array<?xf32>>, index) -> fir.ref<f32>
    %b_val = fir.load %b_elt : fir.ref<f32>
    fir.result %b_elt
  }
  %extent = hlfir.get_extent %vshape, 0 : (fir.shape<1>) -> index
  %c1 = arith.constant 1 : index
  hlfir.forall (%i from %c1 to %extent step %c1) {
    %b_section_val = hlfir.apply %bsection, %i : (hlfir.expr<?xf32>, index) -> f32
    %v_elt = hlfir.designate %v#0, %i : (!fir.box<!fir.array<?xi32>>, index) -> fir.ref<i32>
    %v_val = fir.load %v_elt : fir.ref<i32>
    %cast = fir.convert %v_val : (i32) -> index
    %a_elt = hlfir.designate %a#0, %v_val : (!fir.ref<!fir.array<?xf32>>, index) -> fir.ref<f32>
    hlfir.assign %b_section_val to %a_elt  : f32, fir.ref<f32>
  }
  return
}
```

This would then be lowered as described in the examples above (hlfir.elemental
will be inlined, hlfir.forall will be rewritten into normal loops taking into
account the alias analysis, and hlfir.assign/hlfir.designate operations will be
lowered to fir.array_coor and fir.store operations).

# Alternatives that were not retained

## Using a non-MLIR based mutable CFG representation

An option would have been to extend the PFT to describe expressions in a way
that can be annotated and modified with the ability to introduce temporaries.
This has been rejected because this would imply a whole new set of
infrastructure and data structures while FIR is already using MLIR
infrastructure, so enriching FIR seems a smoother approach and will benefit from
the MLIR infrastructure experience that was gained.

## Using symbols for HLFIR variables

### Using attributes as pseudo variable symbols

Instead of restricting the memory types an HLFIR variable can have, it was
force the defining operation of HLFIR variable SSA values to always be
retrievable. The idea was to add a fir.ref attribute that would repeat the name
of the HLFIR variable. Using such an attribute would prevent MLIR from merging
two operations using different variables when merging IR blocks. (which is the
main reason why the defining op may become inaccessible). The advantage of
forcing the defining operation to be retrievable is that it allowed all Fortran
information of variables (like attributes) to always be accessible in HLFIR
when looking at their uses, and avoids requiring the introduction of fir.box
usages for simply contiguous variables. The big drawback is that this implies
naming all HLFIR variables, and there are many more of them than there are
Fortran named variables. Naming designators with unique names was not very
natural, and would make designator CSE harder. It also made inlining harder,
because inlining HLFIR code without any fir.def/fir.ref attributes renaming
would break the name uniqueness, which could lead to some operations using
different variables to be merged, and to break the assumption that parent
operations must be visible. Renaming would be possible, but would increase
complexity and risks. Besides, inlining may not be the only transformation
doing code motion, and whose complexity would be increased by the naming
constraints.


### Using MLIR symbols for variables

Using MLIR symbols for HLFIR variables has been rejected because MLIR symbols
are mainly intended to deal with globals and functions that may refer to each
other before being defined. Their processing is not as light as normal values,
and would require to turn every FIR operation with a region into an MLIR symbol
table. This would especially be annoying since fir.designator also produces
variables with their own properties, which would imply creating a lot of MLIR
symbols. All the operations that both accept variable and expression operands
would also either need to be more complex in order to both accept SSA values or
MLIR symbol operands (or some fir.as_expr %var operation should be added to
turn a variable into an expression). Given all variable definitions will
dominate their uses, it seems better to use an SSA model with named attributes.
Using SSA values also makes the transition and mixture with lower-level FIR
operations smoother: a variable SSA usage can simply be replaced by lower-level
FIR operations using the same SSA value.

## Using some existing MLIR dialects for the high-level Fortran.

### Why not using Linalg dialect?

The linalg dialects offers a powerful way to represent array operations: the
linalg.generic operation takes a set of input and output arrays, a related set
of affine maps to represent how these inputs/outputs are to be addressed, and a
region detailing what operation should happen at each iteration point, given the
input and output array elements. It seems mainly intended to optimize matmul,
dot, and sum.

Issues:

-   The linalg dialect is tightly linked to the tensor/memref concepts that
    cannot represent byte stride based discontinuity and would most likely
    require FIR to use MLIR memref descriptor format to take advantage of it.
-   It is not clear whether all Fortran array expression addressing can be
    represented as semi affine maps. For instance, vector subscripted entities
    can probably not, which may force creating temporaries for the related
    designator expressions to fit in this framework. Fortran has a lot more
    transformational intrinsics than matmul, dot, and sum that can and should
    still be optimized.

So while there may be benefits to use linalg at the optimization level (like
rewriting fir.sum/fir.matmul to a linalg sum, with dialect types plumbing
around the operand and results, to get tiling done by linalg), using it as a
lowering target would not cover all Fortran needs (especially for the non
semi-affine cases).
So using linalg is for now left as an optimization pass opportunity in some
cases that could be experimented.

### Why not using Shape dialect?

MLIR shape dialect gives a set of operations to manipulate shapes. The
shape.meet operation is exactly similar with hlfir.shape_meet, except that it
returns a tensor or a shape.shape.

The main issue with using the shape dialect is that it is dependent on tensors.
Bringing the tensor toolchain in flang for the sole purpose of manipulating
shape is not seen as beneficial given that the only thing Fortran needs is
shape.meet The shape dialect is a lot more complex because it is intended to
deal with computations involving dynamically ranked entity, which is not the
case in Fortran (assumed rank usage in Fortran is greatly limited).

## Using embox/rebox and box as an alternative to fir.declare/hlfir.designate and hlfir.expr/ variable concept

All Fortran entities (*) can be described at runtime by a fir.box, except for
some attributes that are not part of the runtime descriptors (like TARGET,
OPTIONAL or VOLATILE).  In that sense, it would be possible to have
fir.declare, hlfir.designate, and hlfir.associate be replaced by embox/rebox,
and also to have all operation creating hlfir.expr to create fir.box.

This was rejected because this would lack clarity, and make embox/rebox
semantics way too complex (their codegen is already non-trivial), and also
because it would then not really be possible to know if a fir.box is an
expression or a variable when it is an operand, which would make reasoning
harder: this would already imply that expressions have been buffered, and it is
not clear when looking at a fir.box if the value it describe may change or not,
while a hlfir.expr value cannot change, which allows moving its usages more
easily.

This would also risk generating too many runtime descriptors read and writes
that could make later optimizations harder.

Hence, while this would be functionally possible, this makes the reasoning about
the IR harder and would not benefit high-level optimizations.

(*) This not true for vector subscripted variables, but the proposed plan will
also not allow creating vector subscripted variables as the result of a
hlfir.designate. Lowering will deal with the assignment and input IO special
case using hlfir.elemental.
