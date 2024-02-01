<!--===- docs/OpenACC-descriptor-management.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# OpenACC dialect: Fortran descriptor management in the offload data environment

## Overview

This document describes the details of the Fortran descriptor management during the offload execution using OpenACC.

LLVM Flang compiler uses an extended CFI descriptor data structure to represent some Fortran variables along with their characteristics in memory.  For example, the descriptor is used to access dummy assumed shape arrays in a context of a subroutine where the arrays' bounds are not explicit but are rather passed to the subroutine via the descriptor storage.

During the offload execution a variable data (i.e. the memory holding the actual value of the variable) can be transferred between the host and the device using explicit OpenACC constructs.  Fortran language does not expose the descriptor representation to the user, but the accesses of variables with descriptors on the device may still be done using the descriptor data.  Thus, the implementations must implicitly manage the data transfers of the descriptors along with the transfers of the actual values of the variables.

The MLIR OpenACC dialect is language agnostic so that it can work for Fortran and C/C++ with OpenACC.  The dialect should provide means for expressing the logic for data and descriptor management for the offload data environment.

The chapter numbering in this document refers to:

* F202x: Fortran standard J3/23-007r1
* OpenACC: OpenACC specification version 3.3
* OpenMP: OpenMP specification version 5.2

## CFI descriptor structure

Flang represents the data descriptors in memory using `CFI_cdesc_t` layout specified by F202x 18.5.3, i.e. the variable data address is located in its first member `base_addr`.  The standard does not strictly specify the layout of all the members, moreover, the actual size of the structure may be different for different variables (e.g. due to different ranks).  Other compilers may use different data descriptor formats, e.g. the variable data address may be interleaved with the auxiliary members, or the auxiliary data may be operated as a data structure not containing the variable data address.  In this document we will only consider supporting the CFI descriptor format as supported by Flang.

## Runtime behavior for variables with descriptors

### Pointer variables

OpenACC specifies the pointer attachment behavior in OpenACC 2.6.8.  This paragraph applies both to Fortran and C/C++ pointers, and the Fortran specific representation of the POINTER variables representation is not explicitly specified.  There is a single mention of the "Fortran descriptor (dope vector)" in 2.6.4 with regards to the POINTER/ALLOCATABLE members of data structures.  Chapter 2.7.2 describes the behavior of different data clause actions including `attach` and `detach` actions for the pointers.  Finally, chapters 2.7.4-13 describe what actions are taken for each OpenACC data clause.

The spec operates in terms of the attachment counter associated with each pointer in device memory.  This counter is not exposed to the user code explicitly, i.e. there is no standard way to query the attachment counter for a pointer, but there are `acc_attach*` and `acc_detach*` APIs that affect the attachment counter as well as the data clause actions used with OpenACC constructs.

Here is an example to demonstrate the member pointer attachment:

Example:

```Fortran
module types
  type ty1
     real, pointer :: p(:,:)
  end type ty1
end module types
program main
  use types
  use openacc
  type(ty1) :: d
  real, pointer :: t1(:,:)
  nullify(d%p)
  allocate(t1(2,2))
  
  ! 2.7.9:
  ! Allocates the memory for object 'd' on the device.
  ! The descriptor member of 'd' is a NULL descriptor,
  ! i.e. the host contents of the descriptor is copied
  ! verbatim to the device.
  ! The descriptor storage is created on the device
  ! just as part of the object 'd' storage.
  !$acc enter data create(d)
  
  d%p => t1
  
  ! 2.7.7:
  ! Pointer d%p is not present on the device, so copyin
  ! action is performed for the data pointed to by the pointer:
  ! the memory for 'REAL :: (2,2)' array is allocated on the device
  ! and the host values of the array elements are copied to
  ! the allocated device memory.
  ! Then the attach action is performed, i.e. the contents
  ! of the device descriptor of d%p are updated as:
  !   * The base_addr member of the descriptor on the device
  !     is initialized to the device address of the data
  !     that has been initialized during the copyin.
  !   * The auxiliary members of the device descriptor are initialized
  !     from the host values of the corresponding members.
  !   * The attachment counter of 'd%p' is set to 1.
  !$acc enter data copyin(d%p)
  
  ! 2.7.7:
  ! Pointer d%p is already present on the device, so no copyin
  ! action is performed.
  ! The attach action is performed according to 2.6.8:
  ! since the pointer is associated with the same target as
  ! during the previous attachment, only its attachment counter
  ! is incremented to 2.
  !$acc enter data copyin(d%p)
  
  ! 3.2.29:
  ! The detach action is performed. According to 2.7.2 the attachment
  ! counter of d%p is decremented to 1.
  call acc_detach(d%p)
  
  ! 3.2.29:
  ! The detach action is performed. According to 2.7.2 the attachment
  ! counter of d%p is decremented to 0, which initiates an update
  ! of the the device pointer to have the same value as the corresponding
  ! pointer in the local memory.
  ! We will discuss this in more detail below.
  call acc_detach(d%p)
  
  ! The following construct will fail, because the 'd%p' descriptor's
  ! base_addr is now the host address not accessible on the device.
  ! Without the second 'acc_detach' it will work correctly.
  !$acc serial present(d)
  print *, d%p(1,2)
  !$acc end serial
```

Let's discuss in more detail what happens during the second `acc_detach`. 

OpenACC 2.6.4:

> 1360 An attach action updates the pointer in device memory to point to the device copy of the data
> 1361 that the host pointer targets; see Section 2.7.2. For Fortran array pointers and allocatable arrays,
> 1362 this includes copying any associated descriptor (dope vector) to the device copy of the pointer.
> 1363 When the device pointer target is deallocated, the pointer in device memory should be restored
> 1364 to the host value, so it can be safely copied back to host memory. A detach action updates the
> 1365 pointer in device memory to have the same value as the corresponding pointer in local memory;
> 1366 see Section 2.7.2.

It explicitly says that the associated descriptor copy happens during the attach action, but it does not specify the same for the detach action.  So one interpretation of this could be that only the `base_addr` member is updated, but this would allow chimera descriptors in codes like this:

Example:

```Fortran
  !$acc enter data copyin(d)
  d%p => t1
  !$acc enter data copyin(d%p)
  d%p(10:,10:) => d%p
  call acc_detach(d%p)
  !$acc exit data copyout(d)
  print *, lbound(d%p)
```

At the point of `acc_detach` the host descriptor of `d%p` points to `t1` data and has the lower bounds `(10:, 10:)`, so if the detach action only updates the `base_addr` member of the device descriptor and does not update the auxiliary members from their current host values, then during the `copyout(d)` the host descriptor `d%p` will have the stale lower bounds `(2:, 2:)`.

So the proposed reading of the spec here is that the same note about the descriptor applies equally to the attach and the detach actions.

#### "Moving target"

According to OpenACC 2.6.8:

> 1535 when the pointer is allocated in device memory. **The attachment counter for a pointer is set to one**
> **1536 whenever the pointer is attached to new target address**, and incremented whenever an attach action
> 1537 for that pointer is performed for the same target address.

This clearly applies to the following example, where the second attach action is executed while the host pointer is attached to new target address comparing to the first attach action:

Example:

```Fortran
  !$acc enter data copyin(d)
  !$acc enter data copyin(t1, t2)
  d%p => t1
  !$acc enter data attach(d%p)
  d%p => t2
  !$acc enter data attach(d%p)
```

The spec is not too explicit about the following example, though:

Example:

```Fortran
  !$acc enter data copyin(d)
  !$acc enter data copyin(t1, t2)
  d%p => t1
  !$acc enter data attach(d%p)
  d%p(10:,10:) => d%p
  !$acc enter data attach(d%p)
```

The `d%p` pointer has not been attached to **new target address** between the two attach actions, so the device descriptor update is not strictly required, but the proposed spec reading is to execute the update as if the pointer has changed its association between the two attach actions (i.e. the attachment counter of `d%p` is 1 after the second attach, not 2; and the device descriptor is updated with the current values).

In other words, the Fortran pointer in the context of the attach actions should be considered not just as the target address but as a combination of all the values embedded inside its Fortran descriptor.

#### Pointer dummy arguments

All the same rules apply to the pointer dummy arguments even though OpenACC spec, again, mentions the descriptor copying only in the context of member pointers.  The following test demonstrates that a consistent implementation should apply 2.6.4 for pointer dummy arguments, otherwise, OpenACC programs may exhibit unexpected behavior after seemingly straightforward subroutine inlining/outlining:

Example:

```Fortran
  type(ty1) :: d
  real, pointer :: t1(:,:)
  allocate(t1(2,2))
  call wrapper(d, d%p, t1)
contains
  subroutine wrapper(d, p, t1)
    type(ty1), target :: d
    real, pointer :: p(:,:)
    real, pointer :: t1(:,:)
    !$acc enter data copyin(d)
    !$acc enter data copyin(t1)
    p => t1
    !$acc enter data copyin(p)
    !$acc serial present(d)
    print *, d%p(1,2)
    !$acc end serial
```

If the descriptor contents is not copied during the attach action implied by `copyin(p)`, then this code does not behave the same way as:

```Fortran
  type(ty1) :: d
  real, pointer :: t1(:,:)
  allocate(t1(2,2))
  !$acc enter data copyin(d)
  !$acc enter data copyin(t1)
  d%p => t1
  !$acc enter data copyin(d%p)
  !$acc serial present(d)
  print *, d%p(1,2)
  !$acc end serial
  !call wrapper(d, d%p, t1)
```

#### The descriptor storage allocation

For POINTER members of aggregate variables the descriptor storage is allocated on the device as part of the allocation of the aggregate variable, which is done either explicitly in the user code or implicitly due to 2.6.2:

> 1292 On a compute or combined construct, if a variable appears in a reduction clause but no other
> 1293 data clause, it is treated as if it also appears in a copy clause. Otherwise, for any variable, the
> 1294 compiler will implicitly determine its data attribute on a compute construct if all of the following
> 1295 conditions are met:
> ...
> 1299 An aggregate variable will be treated as if it appears either:
> 1300 • In a present clause if there is a default(present) clause visible at the compute con
> 1301 struct.
> 1302 • In a copy clause otherwise.

Example:

```Fortran
  type(ty1) :: d
  real, target :: t1(2,2)
  d%p => t1
  !$acc enter data copyin(d%p)
  !$acc serial present(d%p)
  print *, d%p(1,2)
  !$acc end serial
```

Due to `d%p` reference in the `present` clause of the `serial` region, the compiler must produce an implicit copy of `d`. In order for the `d%p` pointer attachment to happen the descriptor storage must be created before the attachment happens, so the following order of the clauses must be implied:

```Fortran
  !$acc serial copy(d) present(d%p)
```

In the case of POINTER dummy argument, if the descriptor storage is not explicitly created in the user code, the pointer attachment may not happen due to 2.7.2:

> 1693 If the pointer var is in shared memory or is not present in the current device memory, or if the
> 1694 address to which var points is not present in the current device memory, no action is taken. 

Example:

```Fortran
  d%p => t1
  !$acc enter data copyin(t1)
  call wrapper(d%p)
contains
  subroutine wrapper(p)
    real, pointer :: p(:,:)
    !$acc serial attach(p)
    print *, p(1,2)
    !$acc end serial
```

### Allocatable variables

OpenACC 2.6.4 names both POINTER and ALLOCATABLE members of data structures as *pointer*, so the same attachment rules apply to both, including the case of dummy ALLOCATABLE arguments:

Example:

```Fortran
module types
  type ty2
     real, allocatable :: a(:,:)
  end type ty2
end module types
  use types
  type(ty2), target :: dd
  dd%a = reshape((/1,2,3,4/),(/2,2/))
  call wrapper(dd, dd%a)
contains
  subroutine wrapper(dd, a)
    type(ty2), target :: dd
    real, allocatable, target :: a(:,:)
    !$acc enter data copyin(dd)
    !$acc enter data copyin(a)
    !$acc serial present(dd)
    print *, dd%a(1,2)
    !$acc end serial
```

### Other variables

F18 compiler also uses descriptors for assumed-shape, assumed-rank, polymorphic, ... variables.  The OpenACC specification does not prescribe how an implementation should manage the descriptors for such variables.  In many (all?) cases the descriptors of these variables have a local scope of a single subprogram, and if a descriptor of such a variable is created on the device, then its live range must be limited on the device by the invocation of the subprogram (with any OpenACC constructs inside it).

For example:

```Fortran
  type(ty2), target :: dd
  ...
  call wrapper(dd%a)
contains
  subroutine wrapper(a)
    real :: a(10:,10:)
    !$acc serial copyin(a)
    print *, a(10,11)
    !$acc end serial
```

The dummy assumed-shape argument `a` is represented with a descriptor, which has no storage overlap with `dd%a`, i.e. it is a temporary descriptor created to represent the data `dd%a` in a shape according to the declaration of the dummy argument `a`.  The implementation is not strictly required to transfer all the values embedded inside the descriptor for `a` to the device.  The only required actions for this code are the ones prescribed by the `copyin(a)` clause in 2.7.7.

### Summary

Pointer attachment for POINTER and ALLOCATABLE variables is a "composite" runtime action that involves the following:

* Getting the device address corresponding to the device copy of the descriptor.
* Comparing the current host descriptor contents with the device descriptor contents (for proper attachment counter updates).
* Getting the device address corresponding to the device copy of the data pointed to by the descriptor.
* Copying data from the host to the device to update the device copy of the descriptor: this data may include the device address of the data, the descriptor data describing the element size, dimensions, etc.
* Descriptors with an F18 addendum may also require mapping the data pointed to by the addendum pointer(s) and attaching this pointer(s) into the device copy of the descriptor.

## Representing pointer attachment in MLIR OpenACC dialect

The Fortran pointer attachment logic specified by OpenACC is not trivial, and in order to be expressed in a language independent MLIR OpenACC dialect we propose to use recipes for delegating the complexity of the implementation to F18 runtime.

```Fortran
  !$acc enter data attach(d%p)
```

The frontend generates an `acc.attach` data operation with `augPtr` being an address of the F18 descriptor representing a POINTER/ALLOCATABLE variable.  Note that `augPtr` refers to an abstract augmented pointer structure, which is handled in a language specific manner by the code provided by the `attachRecipe` reference.

The `attachRecipe` is a callback that takes `varPtr` and `augPtr` pointers, and the section's `offset` and `size` computed from the `bounds` operand of `acc.attach`.  Fortran FE passes these arguments directly to F18 runtime that is aware of the descriptor structure and does all the required checks and device memory updates for the device copy of the descriptor, including the attachment counters updates.

```
acc.attach.recipe @attach_ref :
    (!fir.ref<none>, !fir.ref<!fir.box<none>>, index, index) {
^bb0(%base_addr_val : !fir.ref<none>,
     %aug_ptr : !fir.ref<!fir.box<none>>,
     %offset : index,
     %size : index):
  fir.call _FortranAOpenACCAttachDescriptor(%aug_ptr, %base_addr_val, %offset, %size) :
      (!fir.ref<none>, !fir.ref<!fir.box<none>>, index, index) -> none
  acc.yield
}

%descref = hlfir.designate %0#0{"p"}
    {fortran_attrs = #fir.var_attrs<pointer>} :
    (!fir.ref<!fir.type<_QMtypesTty1{p:!fir.box<!fir.ptr<!fir.array<?x?xf32>>>}>>) ->
    !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
%descval = fir.load %descref : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
%base_addr_val = fir.box_addr %descval : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>) ->
    !fir.ptr<!fir.array<?x?xf32>>
%attach_op = acc.attach
    varPtr(%base_addr_val : !fir.ptr<!fir.array<?x?xf32>>)
    augPtr(%descref : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>)
    bounds(...)
    attachRecipe(@attach_ref) ->
    !fir.ptr<!fir.array<?x?xf32>> {name = "d%p", structured = false}
acc.enter_data dataOperands(%attach_op : !fir.ptr<!fir.array<?x?xf32>>)
```

> Note that for languages not using augmented pointers, we can still use `varPtrPtr` operand to represent the "simple" pointer attachment.  The recipe should be omitted in this case.

For other data clauses there is an implied ordering that the data action happens before the attachment:

```Fortran
  !$acc enter data copyin(d%p)
```

```
%copyin_op = acc.copyin
    varPtr(%base_addr_val : !fir.ptr<!fir.array<?x?xf32>>)
    augPtr(%descref : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>)
    bounds(...)
    attachRecipe(@attach_ref) ->
    !fir.ptr<!fir.array<?x?xf32>> {name = "d%p", structured = false}
```

Here, the `copyin` of the data is followed by the pointer attachment.

### F18 runtime support

The `OpenACCAttachDescriptor` API is defined like this:

```C++
/// Implement OpenACC attach semantics for the given Fortran descriptor.
/// The \p data_ptr must match the descriptor's base_addr member value,
/// it is only used for verification.
/// The given \p offset and \p size specify an array section starting
/// offset and the size of the contiguous section for the array cases,
/// e.g. 'attach(array(2:3))'. For scalar cases, the offset must be 0,
/// and the size must match the scalar size.
///
/// TODO: The API needs to take the device id.
void RTNAME(OpenACCAttachDescriptor)(const Descriptor *descriptor_ptr,
                                     const void *data_ptr,
                                     std::size_t offset,
                                     std::size_t size,
                                     const char *sourceFile,
                                     int sourceLine);
```

The implementation's behavior may be described as (OpenACC 2.7.2):

* If the data described by the host address `data_ptr`, `offset` and `size` is not present on the device, RETURN.
* If the data described by `descriptor_ptr` and the descriptor size is not present on the device, RETURN.
* If the descriptor's attachment counter is not 0 and the host descriptor contents matches the host descriptor contents used for the previous attachment, then increment the attachment counter and RETURN.
* Update descriptor on the device:
  * Copy the host descriptor contents to device memory.

  * Copy the device address corresponding to `data_ptr` into the `base_addr` member of the descriptor in device memory.

  * Perform an appropriate data action for all auxiliary pointers, e.g. `present(addendum_ptr)`/`copyin(addendum_ptr[:size])`, and copy the corresponding device addresses into their locations in the descriptor in device memory.

  * Set the descriptor's attachment counter to 1.

* RETURN

All the "is-present" checks and the data actions for the auxiliary pointers must be performed atomically with regards to the present counters bookkeeping.

The API relies on the primitives provided by `liboffload`, so it is provided by a new F18 runtime library, e.g. `FortranOffloadRuntime`, that depends on `FortranRuntime` and `liboffload`.  The F18 driver adds `FortranOffloadRuntime` for linking under `-fopenacc`/`-fopenmp` (and maybe additional switches like `-fopenmp-targets`).

# TODOs:

* Cover the detach action.
