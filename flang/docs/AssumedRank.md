<!--===- docs/AssumedRank.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->
# Assumed-Rank Objects

An assumed-rank dummy data object is a dummy argument that takes its rank from
its effective argument. It is a dummy argument, or the associated entity of a
SELECT RANK in the `RANK DEFAULT` block. Its rank is not known at compile
time. The rank can be anything from 0 (scalar) to the maximum allowed rank in
Fortran (currently 15 according to Fortran 2018 standard section 5.4.6 point
1).

This document summarizes the contexts where assumed-rank objects can appear,
and then describes how they are implemented and lowered to HLFIR and FIR. All
section references are made to the Fortran 2018 standard.

## Fortran Standard References

Here is a list of sections and constraints from the Fortran standard involving
assumed-ranks.

- 7.3.2.2 TYPE
	- C711
- 7.5.6.1 FINAL statement
	- C789
- 8.5.7 CONTIGUOUS attribute
	- C830
- 8.5.8 DIMENSION attribute
- 8.5.8.7 Assumed-rank entity
	- C837
	- C838
	- C839
- 11.1.10 SELECT RANK
- 15.5.2.13 Restrictions on entities associated with dummy arguments
	- 1 (3) (b) and (c)
	- 1 (4) (b) and (c)
- 15.5.2.4 Ordinary dummy variables - point 17
- 18 Interoperability with C
	- 18.3.6 point 2 (5)

### Summary of the constraints:

Assumed-rank can:
- be pointers, allocatables (or have neither of those atttributes).
- be monomorphic or polymorphic (both `TYPE(*)` and `CLASS(*)`)
- have all the attributes, except VALUE and CODIMENSION (C837). Notably, they
  can have the CONTIGUOUS or OPTIONAL attributes (C830).
- appear as an actual argument of an assumed-rank dummy (C838)
- appear as the selector of SELECT RANK (C838)
- appear as the argument of C_LOC and C_SIZEOF from ISO_C_BINDING (C838)
- appear as the first argument of inquiry intrinsic functions (C838). These
  inquiry functions listed in table 16.1 are detailed in the "Assumed-rank
  features" section below.
- appear in BIND(C) and non BIND(C interface (18.1 point 3)
- be finalized on entry as INTENT(OUT) under some conditions that prevents the
  assumed-rank to be associated with an assumed-size.
- be associated with any kind of scalars and arrays, including assumed-size.

Assumed-rank cannot:
- be coarrays (C837)
- have the VALUE attribute (C837)
- be something that is not a named variable (they cannot be the result of a
  function or a component reference)
- appear in a designator other than the case listed above (C838). Notably, they
  cannot be directly addressed, they cannot be used in elemental operations or
  transformational intrinsics, they cannot be used in IO, they cannot be
  assigned to....
- be finalized on entry as INTENT(OUT) if it could be associated with an
  assumed-size (C839).
- be used in a reference to a procedure without an explicit interface
  (15.4.2.2. point 3 (c)).

With regard to aliasing, assumed-rank dummy objects follow the same rules as
for assumed shapes, with the addition of 15.5.2.13 (c) which adds a rule when
the actual is a scalar (adding that TARGET assumed-rank may alias if the actual
argument is a scalar even if they have the CONTIGUOUS attribute, while it is OK
to assume that CONTIGUOUS TARGET assumed shape do not alias with other
dummies).

---

## Assumed-Rank Representations in Flang

### Representation in Semantics
In semantics (there is no concept of assumed-rank expression needed in
`evaluate::Expr`). Such symbols have either `semantics::ObjectEntityDetails` (
dummy data objects) with a `semantics::ArraySpec` that encodes the
"assumed-rank-shape" (can be tested with IsAssumedRank()), or they have
`semantics::AssocEntityDetails` (associated entity in the RANK DEFAULT case).

Inside a select rank, a `semantics::Symbol` is created for the associated
entity with `semantics::AssocEntityDetails` that points to the the selector
and holds the rank outside of the RANK DEFAULT case.

Assumed-rank dummies are also represented in the
`evaluate::characteristics::TypeAndShape` (with the AssumedRank attribute) to
represent assumed-rank in procedure characteristics.

### Runtime Representation of Assumed-Ranks
Assumed-ranks are implemented as CFI_cdesc_t (18.5.3) with the addition of an
f18 specific addendum when required for the type. This is the usual f18
descriptor, and no changes is required to represent assumed-ranks in this data
structure. In fact, there is no difference between the runtime descriptor
created for an assumed shape and the runtime descriptor created when the
corresponding entity is passed as an assumed-rank.

This means that any descriptor can be passed to an assumed-rank dummy (with
care to ensure that the POINTER/ALLOCATABLE attribute match the dummy argument
attributes as usual). Notably, any runtime interface that takes descriptor
arguments of any ranks already work with assumed-rank entities without any
changes or special cases.

This also implies that the runtime cannot tell that an entity is an
assumed-rank based on its descriptor, but there seems to be not need for this
so far ("rank based" dispatching for user defined assignments and IO is not
possible with assumed-ranks, and finalization is possible, but there is no need
for the runtime to distinguish between finalization of an assumed-rank and
finalization of other entities: only the runtime rank matters).

The only difference and difficulty is that descriptor storage size of
assumed-rank cannot be precisely known at compile time, and this impacts the
way descriptor copies are generated in inline code. The size can still be
maximized using the maximum rank, which the runtime code already does when
creating temporary descriptor in many cases. Inline code also needs care if it
needs to access the descriptor addendum (like the type descriptor), since its
offset will not be a compile time constant as usual.

Note that an alternative to maximizing the allocation of assumed-rank temporary
descriptor could be to use automatic allocation based on the rank of the input
descriptor, but this would make stack allocation analysis more complex (tools
will likely not have the Fortran knowledge that this allocation size is bounded
for instance) while the stack "over" allocation is likely reasonable (24 bytes
per dimension). Hence the selection of the simple approach using static size
allocation to the maximum rank.

### Representation in FIR and HLFIR
SSA values for assumed-rank entities have an MLIR type containing a
`!fir.array<*xT>` sequence type wrapped in a `!fir.box` or `!fir.class` type
(additionally wrapped in a `!fir.ref` type for pointers and allocatables).

Examples:
`INTEGER :: x(..)`  -> `!fir.box<!fir.array<* x i32>>` 
`CLASS(*) :: x(..)`  -> `!fir.class<!fir.array<* x none>>`
`TYPE(*) :: x(..)`  -> `!fir.box<!fir.array<* x none>>`
`REAL, ALLOCATABLE :: x(..)`  -> `!fir.ref<!fir.box<!fir.heap<!fir.array<* x f32>>>>`
`TYPE(t), POINTER :: x(..)`  -> `!fir.ref<!fir.box<!fir.ptr<!fir.array<* x !fir.type<t>>>>>` 

All these FIR types are implemented as the address of a CFI_cdesc_t in code
generation.

There is no need to allow assumed-rank "expression" in HLFIR (hlfir.expr) since
assumed-rank cannot appear in expressions (except as the actual argument to an
assumed-rank dummy). Assumed-rank are variables. Also, since they cannot have
the VALUE attribute, there is no need to use the hlfir.as_expr +
hlfir.associate idiom to make copies for them.

FIR/HLFIR operation where assumed-rank may appear:
- as `hlfir.declare` and `fir.declare` operand and result.
- as `fir.convert` operand and/or result.
- as `fir.load` operand and result (POINTER and ALLOCATABLE dereference).
- as a block argument (dummy argument).
- as `fir.rebox_assumed_rank` operand/result (new operation to change some
  fields of assumed-rank descriptors).
- as `fir.box_rank` operand (rank inquiry).
- as `fir.box_dim` operand (brutal user inquiry about the bounds of an
  assumed-rank in a compile time constant dimension).
- as `fir.box_addr` operand (to get the base address in inlined code for
  C_LOC).
- as `fir.box_elesize` operand (to implement LEN and STORAGE_SIZE).
- as `fir.absent` result (passing absent actual to OPTIONAL assumed-rank dummy)
- as `fir.is_present` operand (PRESENT inquiry)
- as `hlfir.copy_in` and `hlfir.copy_out` operand and result (copy in and
  copy-out of assumed-rank)
- as `fir.alloca` type and result (when creating an assumed-rank POINTER dummy
  from a non POINTER dummy).
- as `fir.store` operands (same case as `fir.alloca`).

FIR/HLFIR Operations that should not need to accept assumed-ranks but where it
could still be relevant:
- `fir.box_tdesc` and `fir.box_typecode` (polymorphic assumed-rank cannot
  appear in a SELECT TYPE directly without using a SELECT RANK). Given the
  CFI_cdesc_t structure, no change would be needed for `fir.box_typecode` to
  support assumed-ranks, but `fir.box_tdesc` would require change since the
  position of the type descriptor pointer depends on the rank.
- as `fir.allocmem` / `fir.global` result (assumed-ranks are never local/global
  entities). 
- as `fir.embox` result (When creating descriptor for an explicit shape, the
  descriptor can be created with the entity rank, and then casted via
`fir.convert`).

It is not expected for any other FIR or HLFIR operations to handle assumed-rank
SSA values.

#### Summary of the impact in FIR
One new operation is needed, `fir.rebox_assumed_rank`, the rational being that
fir.rebox codegen is already quite complex and not all the aspects of fir.rebox
matters for assumed-ranks (only simple field changes are required with
assumed-ranks). Also, this operation will be allowed to take an operand in
memory to avoid expensive fir.load of pointer/allocatable inputs. The operation
will also allow creating rank-one assumed-size descriptor from an input
assumed-rank descriptor to cover the SELECT RANK `RANK(*)` case.

It is proposed that the FIR descriptor inquiry operation (fir.box_addr,
fir.box_rank, fir.box_dim, fir.box_elesize at least) be allowed to take
fir.ref<fir.box> arguments (allocatable and pointer descriptors) directly
instead of generating a fir.load first. A conditional "read" effect will be
added in such case. Again, the purpose is to avoid generating descriptor copies
for the sole purpose of satisfying the SSA IR constraints. This change will
likely benefit the non assumed-rank case too (even though LLVM is quite good at
removing pointless descriptor copies in these cases).

It will be ensured that all the operation listed above accept assumed-rank
operands (both the verifiers and coedgen). The codegen of `fir.load`,
`fir.alloca`, `fir.store`, `hlfir.copy_in` and `hlfir.copy_out` will need
special handling for assumed-ranks.

### Representation in LLVM IR

Assumed-rank descriptor types are lowered to the LLVM type of a CFI_cdesc_t
descriptor with no dimension array field and no addendum. That way, any inline
code attempt to directly access dimensions and addendum with constant offset
will be invalid for more safety, but it will still be easy to generate LLVM GEP
to address the first descriptor fields in LLVM (to get the base address, rank,
type code and attributes).

`!fir.box<!fir.array<* x i32>>` -> `!llvm.struct<(ptr, i64, i32, i8, i8, i8, i8>`

## Assumed-rank Features

This section list the different Fortran features where assumed-rank objects are
involved and describes the related implementation design.

### Assumed-rank in procedure references
Assumed-rank arguments are implemented as being the address of a CFI_cdesc_t.

When passing an actual argument to an assumed-rank dummy, the following points
need special attention and are further described below:
- Copy-in/copy-out when required
- Creation of forwarding of the assumed-rank dummy descriptor (including when
  the actual is an assumed-size).
- Finalization, deallocation, and initialization of INTENT(OUT) assumed-rank
  dummy.

OPTIONAL assumed-ranks are implemented like other non assumed-rank OPTIONAL
objects passed by descriptor: an absent assumed-rank is represented by a null
pointer to a CFI_cdesc_t.

The passing interface for assumed-rank described above and below is compliant
by default with the BIND(C) case, except for the assumed-rank dummy descriptor
lower bounds, which are only set to zeros in BIND(C) interface because it
implies in most of the cases to create a new descriptor.

VALUE is forbidden for assumed-rank dummies, so there is nothing to be done for
it (although since copy-in/copy-out is possible, the compiler must anyway deal
with creating assumed-rank copies, so it would likely not be an issue to relax
this constraint).

#### Copy-in and Copy out
Copy-in and copy-out is required when passing an actual that is not contiguous
to a non POINTER CONTIGUOUS assumed-rank.

When the actual argument is ranked, the copy-in/copy-out can be performed on
the ranked actual argument where the dynamic type has been aligned with the
dummy type if needed (passing CLASS(T) to TYPE(T)) as illustrated below.

```Fortran
module m
type t
 integer :: i
end type
contains
subroutine foo(x)
 class(t) :: x(:)
 interface
  subroutine bar(x)
    import :: t
    type(t), contiguous :: x(..)
  end subroutine
 end interface
 ! copy-in and copy-out is required aroud bar
 call bar(x)
end
end module
```

When the actual is also an assumed-rank special the same copy-in/copy-out need
may arise, and the `hlfir.copy_in` and `hlfir.copy_out` are also used to cover
this case. The `hlfir.copy_in`operation is implemented using the `IsContiguous`
runtime (can be used as-is) and the `AssignTemporary` temporary runtime.

The difference with the ranked case is that more care is needed to create the
output descriptor passed to `AssignTemporary`: it must be allocated to the
maximum rank with the same type as the input descriptor and only the descriptor
fields prior to the array dimensions will be initialized to those of an
unallocated descriptor prior to the runtime call (`AssignTemporary` copies the
addendum if needed).

```Fortran
subroutine foo2(x)
 class(t) :: x(..)
 interface
  subroutine bar(x)
    import :: t
    type(t), contiguous :: x(..)
  end subroutine
 end interface
 ! copy-in and copy-out is required aroud bar
 call bar(x)
end
```
#### Creating the descriptor for assumed-rank dummies

There are four cases to distinguish:
1. Actual does not have a descriptor (and is therefore ranked)
2. Actual has a descriptor that can be forwarded for the dummy
3. Actual has a ranked descriptor that cannot be forwarded for the dummy
4. Actual has an assumed-rank descriptor that cannot be forwarded for the dummy

For the first case, a descriptor will be created for the dummy with `fir.embox`
has if it has the rank of the actual argument. This is the same logic as when
dealing with assumed shape or INTENT(IN) POINTER dummy arguments, except that
an extra cast to the assumed-rank descriptor type is added (no-op at runtime).
Care must be taken to set the final dimension extent to -1 in the descriptor
created for an assumed-size actual argument. Note that the descriptor created
for an assumed-size still has the rank of the assumed-size, a rank-one
descriptor will be created for it if needed in a RANK(*) block (nothing says
that an assumed-size should be passed as a rank-one array in 15.5.2.4 point 17).

For the second case, a cast is added to assumed-rank descriptor type if it is
not one already and the descriptor is forwarded.

For the third case, a new ranked descriptor with the dummy attribute/lower
bounds is created from the actual argument descriptor with `fir.rebox` as it is
done when passing to an assume shape dummy, and a cast to the assumed-rank
descriptor is added .

The last case is the same as the third one, except the that the descriptor
manipulation is more complex since the storage size of the descriptor is
unknown. `fir.rebox` codegen is already quite complex since it deals with
creating descriptor for descriptor based array sections and pointer remapping.
Both of those are meaningless in this case where the output descriptor is the
same as the input one, except for the lower bounds, attribute, and derived type
pointer field that may need to be changed to match the values describing the
dummy. A simpler `fir.rebox_assumed_rank` operation is added for this use case.
Notably, this operation can take fir.ref<fir.box> inputs to avoid creating an
expensive and useless fir.load of POINTER/ALLOCATABLE descriptors.

Fortran requires the compiler to fall in the 3rd and 4th case and create
descriptor temporary for the dummy a lot more than one would think and hope. An
annex section below discusses cases that force the compiler to create a new
descriptor for the dummy even if the actual already has a descriptor. These are
the same situations than with non assumed-rank arguments, but when passing
assumed-rank to assumed-ranks, the cost of this extra copy is higher.

#### Intent(out) assumed-rank finalization, deallocation, initialization

The standard prevents INTENT(OUT) assumed-rank requiring finalization to be
associated with assumed-size arrays (C839) because there would be no way to
finalize such entities. But INTENT(OUT) finalization is still possible if the
actual is not an assumed-size and not a nonpointer nonallocatable assumed-rank.

Flang therefore needs to implement finalization, deallocation and
initialization of INTENT(OUT) as usual. Non pointer non allocatable INTENT(OUT)
finalization is done via a call to `Destroy` runtime API that takes a
descriptor and can be directly used with an assumed-rank descriptor with no
change. The initialization is done via a call to the `Initialize` runtime API
that takes a descriptor and can also directly be used with an assumed
descriptor. Conditional deallocation of INTENT(OUT) allocatable is done via an
inline allocation status check and either an inline deallocate for intrinsic
types, or a runtime call to `Deallocate` for the other cases. For
assumed-ranks, the runtime call is always used regardless of the type to avoid
inline descriptor manipulations. `Deallocate` runtime API also works with
assumed-rank descriptors with no changes (like any runtime API taking
descriptors of any rank).

```Fortran
subroutine foo(x)
 class(*), allocatable :: x(..)
 interface
  subroutine bar(x)
    class(*), intent(out) :: x(..)
  end subroutine
 end interface
 ! x may require finalization and initialization on bar entry.
 call bar(x)
end
subroutine bar(x)
  class(*), intent(out) :: x(..)
end subroutine
```
### Select Rank

Select rank is implemented with a rank inquiry (and last extent for `RANK(*)`),
followed by a jump in the related block where the selector descriptor is cast
to a descriptor with the associated entity rank for the current block for the
`RANK(cst)` cases. In the `RANK DEFAULT`, the input descriptor is kept with no
cast, and in the RANK(*), a rank-one descriptor is created with the same
dynamic type as the input.
These new descriptor values are mapped to the associated entity symbol and
lowering precede as usual. This is very similar to how Select Type is
implemented. The `RANK(*)` is a bit odd, it detects assumed-ranks associated
with an assumed-size arrays regardless of the rank, and takes precedence over
any rank based matching.

Note that `-1` is a magic extent number that encodes that a descriptor describes
an entity that is an assumed-size (user specified extents of explicit shape
arrays are always normalized to zero when negative, so `-1` is a safe value to
identify a descriptor created for an assumed-size). It is actually well
specified for the BIND(C) (18.5.2 point 1.) and is always used as such in flang
descriptors.

The implementation of SELECT RANK is done as follow:
- Read the rank `r` in the descriptor
- If there is a `RANK(*)`, read the extent in dimension `r`. If it is `-1`,
  jump to the `RANK(*)` block. Otherwise, continue to the steps below.
- For each `RANK(constant)` case, compare `constant` to `r`. Stop at first
  match and jump to related block. The order of the comparisons does not matter
(there cannot be more than one match).
- Jump to `RANK DEFAULT` block is any. Otherwise jump to the end of the
  construct.

The blocks for each cases jumps at the end of the construct at the end. As
opposed to SELECT TYPE, no clean-up should be needed at the construct level
since the select-rank selector is a named entity and cannot be a temporary with
a lifetime of the construct.

Except for the `RANK(*)` case, the branching logic is implemented in FIR with a
`fir.select_case` operating on the rank.

Example:

```Fortran
subroutine test(x)
  interface
    subroutine assumed_size(x)
      real :: x(*)
    end subroutine
    subroutine scalar(x)
      real :: x
    end subroutine
    subroutine rank_one(x)
      real :: x(:)
    end subroutine
    subroutine many_dim_array(x)
      real :: x(..)
    end subroutine
  end interface
  
  real :: x(..)
  select rank (y => x)
  rank(*)
    call assumed_size(y)
  rank(0)
    call scalar(y)
  rank(1)
    call rank_one(y)
  rank default
    call many_dim_array(y)
  end select
end subroutine
```

Pseudo FIR for the example (some converts and SSA constants creation are not shown for more clarity):

```
func.func @_QPtest(%arg0: !fir.box<!fir.array<?xf32>>) {
  %x:2 = hlfir.declare %arg0 {uniq_name = "_QFtestEx"} : (!fir.box<!fir.array<*xf32>>) -> (!fir.box<!fir.array<*xf32>>, !fir.box<!fir.array<*xf32>>)
  %r = fir.box_rank %x#1 : (!fir.box<!fir.array<*xf32>>) -> i32
  %last_extent = fir.call @_FortranASizeDim(%x#1, %r, %sourcename, %sourceline)
  %is_assumed_size = arith.cmpi eq %last_extent, %c-1: (i64, i64) -> i1
  cf.cond_br %is_assumed_size, ^bb_assumed_size, ^bb_not_assumed_size
^bb_assumed_size:
  %r1_box = fir.rebox_assumed_rank %x#0 : (!fir.box<!fir.array<*xf32>>) -> !fir.box<!fir.array<?xf32>>
  %addr = fir.box_addr %addr, !fir.box<!fir.array<?xf32>> -> !fir.ref<!fir.array<?xf32>>
  fir.call @_QPassumed_size(%addr) (!fir.ref<!fir.array<?xf32>>) -> ()
  cf.br ^bb_end
^bb_not_assumed_size:
  fir.select_case %3 : i32 [#fir.point, %c0, ^bb_scalar, #fir.point, %c1, ^bb_rank1, unit, ^bb_default]
^bb_scalar:
  %scalar_cast = fir.convert %x#1 : (!fir.box<!fir.array<*xf32>>) -> !fir.box<f32>
  %x_scalar = fir.box_addr %scalar_cast: (!fir.box<f32>) -> !fir.ref<f32>
  fir.call @_QPscalar(%x_scalar) (!fir.ref<f32>) -> ()
  cf.br ^bb_end
^bb_rank1:
  %rank1_cast = fir.convert %x#1 : (!fir.box<!fir.array<*xf32>>) -> !fir.box<!fir.array<?xf32>>
  fir.call @_QPrank_one(%rank1_cast) (!fir.box<!fir.array<?xf32>>) -> ()
  cf.br ^bb_end
^bb_default:
  fir.call @_QPmany_dim_array(%x#1) (!fir.box<!fir.array<*xf32>>) -> ()
  cf.br ^bb_end
^bb_end
  return
}
```

### Inquiry intrinsic functions
#### ALLOCATED and ASSOCIATED
Implemented inline with `fir.box_addr` (reading the descriptor first address
inline). Currently, FIR descriptor inquiry happens at the "descriptor value"
level (require a fir.load of the POINTER or ALLOCATABLE !fir.ref<!fir.box<>>),
to satisfy the SSA value semantics, the fir.load creates a copy of the
underlying descriptor storage. With assume ranks, this copy will be "expensive"
and harder to optimize out given the descriptor storage size is not a compile
time constant. To avoid this extra cost, ALLOCATABLE and POINTER assumed-ranks
will be cast to scalar descriptors before the `fir.load`.

```Fortran
real, allocatable :: x(..)
print *, allocated(x)
```

```
%1 = fir.convert %x : (!fir.ref<!fir.box<!fir.heap<!fir.array<* x f32>>>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
%2 = fir.load %x : !fir.ref<!fir.box<!fir.heap<f32>>>
%addr = fir.box_addr %2 : (!fir.box<!fir.heap<f32>>) -> fir.ref<f32>
# .... "addr != null" as usual
```
#### LEN and STORAGE_SIZE
Implemented inline with `fir.box_elesize` with the same approach as
ALLOCATED/ASSOCIATED when dealing with fir.box load for POINTERS and
ALLOCATABLES.

```Fortran
character(*) :: x(..)
print *, len(x)
```

```
%ele_size = fir.box_elesize %x : (!fir.box<!fir.array<*x!fir.char<?>>>) -> i64
# .... divide by character KIND byte size if needed as usual 
```
#### PRESENT
Implemented inline with `fir.is_present` which ends-up implemented as a check
that the descriptor address is not null just like with OPTIONAL assumed shapes
and OPTIONAL pointers and allocatables.

```Fortran
real, optional :: x(..)
print *, present(x)
```

```
%is_present = fir.is_prent %x : (!fir.box<!fir.array<*xf32>>) -> i1
```
#### RANK
Implemented inline with `fir.box_rank` which simply reads the descriptor rank
field.

```Fortran
real :: x(..)
print *, len(x)
```

```
%rank = fir.box_rank %x : (!fir.box<!fir.array<*xf32>>) -> i32
```
#### SIZE
Using the runtime can be queried as it is done for assumed shapes. When DIM is
present and is constant, `fir.box_dim` can also be used with the option to add
a runtime check that RANK <= DIM. Pointers and allocatables are dereferenced,
which in FIR currently creates a descriptor copy that cannot be simplified
like for the previous inquiries by inserting a cast before the fir.load (the
dimension info must be correctly copied). 

#### LBOUND, SHAPE, and UBOUND
When DIM is present an is present, the runtime can be used as it is currently
with assumed shapes. When DIM is absent, the result is a rank-one array whose
extent is the rank. The runtime has an entry for UBOUND that takes a descriptor
and allocate the result as needed, so the same logic as for assumed shape can
be used.

There is no such entry for LBOUND/SHAPE currently, it would likely be best to
add one rather than to jungle with inline code. Pointers and allocatables
dereference is similar as with SIZE.

#### EXTENDS_TYPE_OF, SAME_TYPE_AS, and IS_CONTIGUOUS
Using the runtime as it is done currently with assumed shapes. Pointers and
allocatables dereference is similar as with SIZE.

#### C_LOC from ISO_C_BINDING
Implemented with `fir.box_addr` as with other C_LOC cases for entities that
have descriptors.

#### C_SIZE_OF from ISO_C_BINDING
Implemented as STORAGE_SIZE * SIZE.

#### Floating point inquiries and NEW_LINE
BIT_SIZE, DIGITS, EPSILON, HUGE, KIND, MAXEXPONENT, MINEXPONENT, NEW_LINE,
PRECISION, RADIX, RANGE, TINY all accept assumed-rank, but are always constant
folded by semantics based on the type and lowering does not need to deal with
them.

#### Coarray inquiries
Assumed-rank cannot be coarrays (C837), but they can still technically appear
in COSHAPE (which should obviously return zero). They cannot appear in LBOUND,
LCOBOUND, UBOUND, UCOBOUND that require the argument to be a coarray.

## Annex 1 - Descriptor temporary for the dummy arguments

When passing an actual argument that is descriptor to a dummy that must be
passed by descriptor, one could expect that the descriptor of the actual can
just be forwarded to the dummy, but this is unfortunately not possible in quite
some cases. This is not specific to assumed-ranks, but since the cost of
descriptor temporaries is higher for assumed-ranked, it is discussed here.

Below are the reasons for which a new descriptor may be required:
1. passing a POINTER to a non POINTER
2. setting the descriptor CFI_cdesc_t `attribute` according to the dummy
   POINTER/ALLOCATABLE attributes (18.3.6 point 4 for the BIND(C) case).
3. setting the CFI_cdesc_t lower bounds to zero for a BIND(C) assumed
   shape/rank dummy (18.5.3 point 3).
4. setting the derived type pointer to the dummy dynamic type when passing a
   CLASS() actual to a TYPE() dummy.

Justification of 1.:
When passing a POINTER to a non POINTER, the target of the pointer is passed,
and nothing prevents the association status of the actual argument to change
during the call (e.g. if the POINTER is another argument of the call, or is a
module variable, it may be re-associated in the call). These association status
change of the actual should not impact the dummy, so they must not share the
same descriptor.

Justification of 2.:
In the BIND(C) case, this is required by 18.3.6 point 4. Outside of the BIND(C)
case, this should still be done because any runtime call where the dummy
descriptor is forwarded may misbehave if the ALLOCATABLE/POINTER attribute is
not the one of the dummy (e.g. reallocation could be triggered instead of
padding/trimming characters).

Justification of 3:
18.5.3 point 3.

Justification of 4:
If the descriptor derived type info pointer is not the one of the dummy dynamic
type, many runtime call like IO and assignment will misbehave when being
provided the dummy descriptor.

For point 2., 3., and 4., one could be tempted to change the descriptor fields
before and after the call, but this is risky since this would assume nothing
will access the actual argument descriptor during the call. And even without
bringing any potential asynchronous behavior of OpenMP/OpenACC/Cuda Fortran
extensions, the actual argument descriptor may be passed inside a call in
another arguments with "different" lower bounds POINTER or ALLOCATABLE (but
could also be accessed via host of use association in general).


## Annex 2 - Assumed-Rank Objects and IGNORE_TKR

It is possible to:
- Set IGNORE_TKR(TK) on assumed-rank dummies (but TYPE(*) is better when
  possible).
- Pass an assumed-rank to an IGNORE_TKR(R) dummy that is not passed
  by descriptor (explicit shape and assumed-size). Note that copy-in and
  copy-out will be performed for the dummy

It is not possible to:
- Set IGNORE_TKR(R) on an assumed-rank dummy.

Example:

```Fortran
subroutine test(assumed_rank_actual)
interface
 subroutine assumed_size_dummy(x)
    !dir$ ignore_tkr(tkr) x
    integer :: x(*)
 end subroutine
 subroutine any_type_assumed_rank(x)
    !dir$ ignore_tkr(tk) x
    integer :: x(..)
 end subroutine
end interface
  real :: assumed_rank_actual(..)
  call assumed_size_dummy(assumed_rank_actual) !OK
  call any_type_assumed_rank(assumed_rank_actual) !OK
end subroutine
```

## Annex 3 - Test Plan

MPI_f08 module makes usage of assumed-rank (see
https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report.pdf).  As such compiling
MPI_f08 modules of MPI libraries and some applications making usage of MPI_f08
will be a good test for the implementation of this feature.
