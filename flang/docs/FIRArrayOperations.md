<!--===- docs/FIRArrayOperations.md 
  
   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  
-->

# Design: FIR Array operations

```eval_rst
.. contents::
   :local:
```

## General

The array operations in FIR model the copy-in/copy-out semantics over Fortran
statements.

Fortran language semantics sometimes require the compiler to make a temporary 
copy of an array or array slice. Situations where this can occur include:

* Passing a non-contiguous array to a procedure that does not declare it as
  assumed-shape.
* Array expressions, especially those involving `RESHAPE`, `PACK`, and `MERGE`.
* Assignments of arrays where the array appears on both the left and right-hand
  sides of the assignment.
* Assignments of `POINTER` arrays.

There are currently the following operations:
- `fir.array_load`
- `fir.array_merge_store`
- `fir.array_fetch`
- `fir.array_update`
- `fir.array_access`
- `fir.array_amend`

`array_load`(s) and `array_merge_store` are a pairing that brackets the lifetime
of the array copies.

`array_fetch` and `array_update` are defined to work as getter/setter pairs on 
values of elements from loaded array copies. These have "GEP-like" syntax and
semantics.

Fortran arrays are implicitly memory bound as are some other Fortran type/kind
entities. For entities that can be atomically promoted to the value domain,
we use `array_fetch` and `array_update`.

`array_access` and `array_amend` are defined to work as getter/setter pairs on
references to elements in loaded array copies. `array_access` has "GEP-like"
syntax. `array_amend` annotates which loaded array copy is being written to.
It is invalid to update an array copy without `array_amend`; doing so will
result in undefined behavior.
For those type/kinds that cannot be promoted to values, we must leave them in a
memory reference domain, and we use `array_access` and `array_amend`.

## array_load

This operation taken with `array_merge_store` captures Fortran's
copy-in/copy-out semantics. One way to think of this is that array_load
creates a snapshot copy of the entire array. This copy can then be used
as the "original value" of the array while the array's new value is
computed. The `array_merge_store` operation is the copy-out semantics, which
merge the updates with the original array value to produce the final array
result. This abstracts the copy operations as opposed to always creating
copies or requiring dependence analysis be performed on the syntax trees
and before lowering to the IR.

Load an entire array as a single SSA value.

```fortran
  real :: a(o:n,p:m)
  ...
  ... = ... a ...
```

One can use `fir.array_load` to produce an ssa-value that captures an
immutable value of the entire array `a`, as in the Fortran array expression
shown above. Subsequent changes to the memory containing the array do not
alter its composite value. This operation lets one load an array as a
value while applying a runtime shape, shift, or slice to the memory
reference, and its semantics guarantee immutability.

```mlir
%s = fir.shape_shift %lb1, %ex1, %lb2, %ex2 : (index, index, index, index) -> !fir.shape<2>
// load the entire array 'a'
%v = fir.array_load %a(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
// a fir.store here into array %a does not change %v
```

# array_merge_store

The `array_merge_store` operation stores a merged array value to memory. 


```fortran
  real :: a(n,m)
  ...
  a = ...
```

One can use `fir.array_merge_store` to merge/copy the value of `a` in an
array expression as shown above.

```mlir
  %v = fir.array_load %a(%shape) : ...
  %r = fir.array_update %v, %f, %i, %j : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
  fir.array_merge_store %v, %r to %a : !fir.ref<!fir.array<?x?xf32>>
```

This operation merges the original loaded array value, `%v`, with the
chained updates, `%r`, and stores the result to the array at address, `%a`.

This operation taken with `array_load`'s captures Fortran's
copy-in/copy-out semantics. The first operands of `array_merge_store` is the
result of the initial `array_load` operation. While this value could be
retrieved by reference chasing through the different array operations it is
useful to have it on hand directly for analysis passes since this directly
defines the "bounds" of the Fortran statement represented by these operations.
The intention is to allow copy-in/copy-out regions to be easily delineated,
analyzed, and optimized.

## array_fetch

The `array_fetch` operation fetches the value of an element in an array value.

```fortran
  real :: a(n,m)
  ...
  ... a ...
  ... a(r,s+1) ...
```

One can use `fir.array_fetch` to fetch the (implied) value of `a(i,j)` in
an array expression as shown above. It can also be used to extract the
element `a(r,s+1)` in the second expression.

```mlir
  %s = fir.shape %n, %m : (index, index) -> !fir.shape<2>
  // load the entire array 'a'
  %v = fir.array_load %a(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
  // fetch the value of one of the array value's elements
  %1 = fir.array_fetch %v, %i, %j : (!fir.array<?x?xf32>, index, index) -> f32
```

It is only possible to use `array_fetch` on an `array_load` result value or a
value that can be trace back transitively to an `array_load` as the dominating
source. Other array operation such as `array_update` can be in between.

## array_update

The `array_update` operation is used to update the value of an element in an
array value. A new array value is returned where all element values of the input
array are identical except for the selected element which is the value passed in
the update.

```fortran
  real :: a(n,m)
  ...
  a = ...
```

One can use `fir.array_update` to update the (implied) value of `a(i,j)`
in an array expression as shown above.

```mlir
  %s = fir.shape %n, %m : (index, index) -> !fir.shape<2>
  // load the entire array 'a'
  %v = fir.array_load %a(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
  // update the value of one of the array value's elements
  // %r_{ij} = %f  if (i,j) = (%i,%j),   %v_{ij} otherwise
  %r = fir.array_update %v, %f, %i, %j : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
  fir.array_merge_store %v, %r to %a : !fir.ref<!fir.array<?x?xf32>>
```

An array value update behaves as if a mapping function from the indices
to the new value has been added, replacing the previous mapping. These
mappings can be added to the ssa-value, but will not be materialized in
memory until the `fir.array_merge_store` is performed.
`fir.array_update` can be seen as an array access with a notion that the array
will be changed at the accessed position when `fir.array_merge_store` is
performed.

## array_access

The `array_access` provides a reference to a single element from an array value.
This is *not* a view in the immutable array, otherwise it couldn't be stored to.
It can be see as a logical copy of the element and its position in the array.
Tis reference can be written to and modified withoiut changing the original
array.

The `array_access` operation is used to fetch the memory reference of an element
in an array value.

```fortran
  real :: a(n,m)
  ...
  ... a ...
  ... a(r,s+1) ...
```

One can use `fir.array_access` to recover the implied memory reference to
the element `a(i,j)` in an array expression `a` as shown above. It can also
be used to recover the reference element `a(r,s+1)` in the second
expression.

```mlir
  %s = fir.shape %n, %m : (index, index) -> !fir.shape<2>
  // load the entire array 'a'
  %v = fir.array_load %a(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
  // fetch the value of one of the array value's elements
  %1 = fir.array_access %v, %i, %j : (!fir.array<?x?xf32>, index, index) -> !fir.ref<f32>
```

It is only possible to use `array_access` on an `array_load` result value or a
value that can be trace back transitively to an `array_load` as the dominating
source. Other array operation such as `array_amend` can be in between.

`array_access` if mainly used with `character`'s arrays and arrays of derived
types where because they might have a non-compile time sizes that would be
useless too load entirely or too big to load. 

Here is a simple example with a `character` array assignment. 

Fortran
```
subroutine foo(c1, c2, n)
  integer(8) :: n
  character(n) :: c1(:), c2(:)
  c1 = c2
end subroutine
```

It results in this cleaned-up FIR:
```
func @_QPfoo(%arg0: !fir.box<!fir.array<?x!fir.char<1,?>>>, %arg1: !fir.box<!fir.array<?x!fir.char<1,?>>>, %arg2: !fir.ref<i64>) {
    %0 = fir.load %arg2 : !fir.ref<i64>
    %c0 = arith.constant 0 : index
    %1:3 = fir.box_dims %arg0, %c0 : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index) -> (index, index, index)
    %2 = fir.array_load %arg0 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.array<?x!fir.char<1,?>>
    %3 = fir.array_load %arg1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.array<?x!fir.char<1,?>>
    %c1 = arith.constant 1 : index
    %4 = arith.subi %1#1, %c1 : index
    %5 = fir.do_loop %arg3 = %c0 to %4 step %c1 unordered iter_args(%arg4 = %2) -> (!fir.array<?x!fir.char<1,?>>) {
      %6 = fir.array_access %3, %arg3 : (!fir.array<?x!fir.char<1,?>>, index) -> !fir.ref<!fir.char<1,?>>
      %7 = fir.array_access %arg4, %arg3 : (!fir.array<?x!fir.char<1,?>>, index) -> !fir.ref<!fir.char<1,?>>
      %false = arith.constant false
      %8 = fir.convert %7 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
      %9 = fir.convert %6 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
      fir.call @llvm.memmove.p0i8.p0i8.i64(%8, %9, %0, %false) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
      %10 = fir.array_amend %arg4, %7 : (!fir.array<?x!fir.char<1,?>>, !fir.ref<!fir.char<1,?>>) -> !fir.array<?x!fir.char<1,?>>
      fir.result %10 : !fir.array<?x!fir.char<1,?>>
    }
    fir.array_merge_store %2, %5 to %arg0 : !fir.array<?x!fir.char<1,?>>, !fir.array<?x!fir.char<1,?>>, !fir.box<!fir.array<?x!fir.char<1,?>>>
    return
  }
  func private @llvm.memmove.p0i8.p0i8.i64(!fir.ref<i8>, !fir.ref<i8>, i64, i1)
}
```

`fir.array_access` and `fir.array_amend` split the two purposes of
`fir.array_update` into two distinct operations to work on type/kind that must
reside in the memory reference domain. `fir.array_access` captures the array
access semantics and `fir.array_amend` denotes which `fir.array_access` is the
lhs.

We do not want to start loading the entire `!fir.ref<!fir.char<1,?>>` here since
it has dynamic length, and even if constant, could be too long to do so.

## array_amend

The `array_amend` operation marks an array value as having been changed via a 
reference obtain by an `array_access`. It acts as a logical transaction log
that is used to merge the final result back with an `array_merge_store`
operation.

```mlir
  // fetch the value of one of the array value's elements
  %1 = fir.array_access %v, %i, %j : (!fir.array<?x?xT>, index, index) -> !fir.ref<T>
  // modify the element by storing data using %1 as a reference
  %2 = ... %1 ...
  // mark the array value
  %new_v = fir.array_amend %v, %2 : (!fir.array<?x?xT>, !fir.ref<T>) -> !fir.array<?x?xT>
```

## Example

Here is an example of a FIR code using several array operations together. The
example below is a simplified version of the FIR code comiing from the
following Fortran code snippet.

```fortran
subroutine s(a,l,u)
  type t
    integer m
  end type t
  type(t) :: a(:)
  integer :: l, u
  forall (i=l:u)
    a(i) = a(u-i+1)
  end forall
end
```

```
func @_QPs(%arg0: !fir.box<!fir.array<?x!fir.type<_QFsTt{m:i32}>>>, %arg1: !fir.ref<i32>, %arg2: !fir.ref<i32>) {
  %l = fir.load %arg1 : !fir.ref<i32>
  %l_index = fir.convert %l : (i32) -> index
  %u = fir.load %arg2 : !fir.ref<i32>
  %u_index = fir.convert %u : (i32) -> index
  %c1 = arith.constant 1 : index
  // This is the "copy-in" array used on the RHS of the expression. It will be indexed into and loaded at each iteration.
  %array_a_src = fir.array_load %arg0 : (!fir.box<!fir.array<?x!fir.type<_QFsTt{m:i32}>>>) -> !fir.array<?x!fir.type<_QFsTt{m:i32}>>

  // This is the "seed" for the "copy-out" array on the LHS. It'll flow from iteration to iteration and gets
  // updated at each iteration.
  %array_a_dest_init = fir.array_load %arg0 : (!fir.box<!fir.array<?x!fir.type<_QFsTt{m:i32}>>>) -> !fir.array<?x!fir.type<_QFsTt{m:i32}>>
  
  %array_a_final = fir.do_loop %i = %l_index to %u_index step %c1 unordered iter_args(%array_a_dest = %array_a_dest_init) -> (!fir.array<?x!fir.type<_QFsTt{m:i32}>>) {
    // Compute indexing for the RHS and array the element.
    %u_minus_i = arith.subi %u_index, %i : index // u-i
    %u_minus_i_plus_one = arith.addi %u_minus_i, %c1: index // u-i+1
    %a_src_ref = fir.array_access %array_a_src, %u_minus_i_plus_one {Fortran.offsets} : (!fir.array<?x!fir.type<_QFsTt{m:i32}>>, index) -> !fir.ref<!fir.type<_QFsTt{m:i32}>>
    %a_src_elt = fir.load %a_src_ref : !fir.ref<!fir.type<_QFsTt{m:i32}>>

    // Get the reference to the element in the array on the LHS
    %a_dst_ref = fir.array_access %array_a_dest, %i {Fortran.offsets} : (!fir.array<?x!fir.type<_QFsTt{m:i32}>>, index) -> !fir.ref<!fir.type<_QFsTt{m:i32}>>

    // Store the value, and update the array
    fir.store %a_src_elt to %a_dst_ref : !fir.ref<!fir.type<_QFsTt{m:i32}>>
    %updated_array_a = fir.array_amend %array_a_dest, %a_dst_ref : (!fir.array<?x!fir.type<_QFsTt{m:i32}>>, !fir.ref<!fir.type<_QFsTt{m:i32}>>) -> !fir.array<?x!fir.type<_QFsTt{m:i32}>>

    // Forward the current updated array to the next iteration.
    fir.result %updated_array_a : !fir.array<?x!fir.type<_QFsTt{m:i32}>>
  }
  // Store back the result by merging the initial value loaded before the loop
  // with the final one produced by the loop.
  fir.array_merge_store %array_a_dest_init, %array_a_final to %arg0 : !fir.array<?x!fir.type<_QFsTt{m:i32}>>, !fir.array<?x!fir.type<_QFsTt{m:i32}>>, !fir.box<!fir.array<?x!fir.type<_QFsTt{m:i32}>>>
  return
}
```
