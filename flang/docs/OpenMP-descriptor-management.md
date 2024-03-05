<!--===- docs/OpenMP-descriptor-management.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# OpenMP dialect: Fortran descriptor type mapping for offload

The initial method for mapping Fortran types tied to descriptors for OpenMP offloading is to treat these types 
as a special case of OpenMP record type (C/C++ structure/class, Fortran derived type etc.) mapping as far as the 
runtime is concerned. Where the box (descriptor information) is the holding container and the underlying 
data pointer is contained within the container, and we must generate explicit maps for both the pointer member and
the container. As an example, a small C++ program that is equivalent to the concept described, with the 
`mock_descriptor` class being representative of the class utilised for descriptors in Clang:

```C++
struct mock_descriptor {
  long int x;
  std::byte x1, x2, x3, x4;
  void *pointer;
  long int lx[1][3];
};

int main() {
mock_descriptor data;
#pragma omp target map(tofrom: data, data.pointer[:upper_bound])
{
    do something... 
}

 return 0;
}
```

In the above, we have to map both the containing structure, with its non-pointer members and the
data pointed to by the pointer contained within the structure to appropriately access the data. This 
is effectively what is done with descriptor types for the time being. Other pointers that are part 
of the descriptor container such as the addendum should also be treated as the data pointer is 
treated.

Currently, Flang will lower these descriptor types in the OpenMP lowering (lower/OpenMP.cpp) similarly
to all other map types, generating an omp.MapInfoOp containing relevant information required for lowering
the OpenMP dialect to LLVM-IR during the final stages of the MLIR lowering. However, after 
the lowering to FIR/HLFIR has been performed an OpenMP dialect specific pass for Fortran, 
`OMPDescriptorMapInfoGenPass` (Optimizer/OMPDescriptorMapInfoGen.cpp) will expand the 
`omp.MapInfoOp`'s containing descriptors (which currently will be a `BoxType` or `BoxAddrOp`) into multiple 
mappings, with one extra per pointer member in the descriptor that is supported on top of the original
descriptor map operation. These pointers members are linked to the parent descriptor by adding them to 
the member field of the original descriptor map operation, they are then inserted into the relevant map
owning operation's (`omp.TargetOp`, `omp.DataOp` etc.) map operand list and in cases where the owning operation
is `IsolatedFromAbove`, it also inserts them as `BlockArgs` to canonicalize the mappings and simplify lowering.

An example transformation by the `OMPDescriptorMapInfoGenPass`:

```

...
%12 = omp.map_info var_ptr(%1#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.box<!fir.ptr<!fir.array<?xi32>>>) map_clauses(tofrom) capture(ByRef) bounds(%11) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {name = "arg_alloc"}
...
omp.target map_entries(%12 -> %arg1, %13 -> %arg2 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<i32>) {
    ^bb0(%arg1: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, %arg2: !fir.ref<i32>):
...

====>

...
%12 = fir.box_offset %1#1 base_addr : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
%13 = omp.map_info var_ptr(%1#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.array<?xi32>) var_ptr_ptr(%12 : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) map_clauses(tofrom) capture(ByRef) bounds(%11) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {name = ""}
%14 = omp.map_info var_ptr(%1#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.box<!fir.ptr<!fir.array<?xi32>>>) map_clauses(tofrom) capture(ByRef) members(%13 : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {name = "arg_alloc"}
...
omp.target map_entries(%13 -> %arg1, %14 -> %arg2, %15 -> %arg3 : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<i32>) {
    ^bb0(%arg1: !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, %arg2: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, %arg3: !fir.ref<i32>):
...

```

In later stages of the compilation flow when the OpenMP dialect is being lowered to LLVM-IR these descriptor
mappings are treated as if they were structure mappings with explicit member maps on the same directive as 
their parent was mapped. 

This implementation utilises the member field of the `map_info` operation to indicate that the pointer 
descriptor elements which are contained in their own `map_info` operation are part of their respective 
parent descriptor. This allows the descriptor containing the descriptor pointer member to be mapped
as a composite entity during lowering, with the correct mappings being generated to tie them together,
allowing the OpenMP runtime to map them correctly, attaching the pointer member to the parent
structure so it can be accessed during execution. If we opt to not treat the descriptor as a single 
entity we have issues with the member being correctly attached to the parent and being accessible,
this can cause runtime segfaults on the device when we try to access the data through the parent. It
may be possible to avoid this member mapping, treating them as individual entities, but treating a 
composite mapping as an individual mapping could lead to problems such as the runtime taking 
liberties with the mapping it usually wouldn't if it knew they were linked, we would also have to 
be careful to maintian the correct order of mappings as we lower, if we misorder the maps, it'd be
possible to overwrite already written data, e.g. if we write the descriptor data pointer first, and
then the containing descriptor, we would overwrite the descriptor data pointer with the incorrect 
address.

This method is generic in the sense that the OpenMP dialect doesn't need to understand that it is mapping a 
Fortran type containing a descriptor, it just thinks it's a record type from either Fortran or C++. However,
it is a little rigid in how the descriptor mappings are handled as there is no specialisation or possibility
to specialise the mappings for possible edge cases without polluting the dialect or lowering with further
knowledge of Fortran and the FIR dialect.

# OpenMP dialect differences from OpenACC dialect

The descriptor mapping for OpenMP currently works differently to the planned direction for OpenACC, however, 
it is possible and would likely be ideal to align the method with OpenACC in the future. 

Currently the OpenMP specification is less descriptive and has less stringent rules around descriptor based
types so does not require as complex a set of descriptor management rules as OpenACC (although, in certain 
cases for the interim adopting OpenACC's rules where it makes sense could be useful). To handle the more 
complex descriptor mapping rules OpenACC has opted to utilise a more runtime oriented approach, where 
specialized runtime functions for handling descriptor mapping for OpenACC are created and these runtime 
function handles are attatched to a special OpenACC dialect operation. When this operation is lowered it 
will lower to the attatched OpenACC descriptor mapping runtime function. This sounds like it will work 
(no implementation yet) similarly to some of the existing HLFIR operations which optionally lower to 
Fortran runtime calls. 

This methodology described by OpenACC which utilises runtime functions to handle specialised mappings allows
more flexibility as a significant amount of the mapping logic can be moved into the runtime from the compiler.
It also allows specialisation of the mapping for fortran specific types. This may be a desireable approach
to take for OpenMP in the future, in particular if we find need to specialise mapping further for 
descriptors or other Fortran types. However, for the moment the currently chosen implementation for OpenMP
appears sufficient as far as the OpenMP specification and current testing can show.
