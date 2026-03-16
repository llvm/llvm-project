! Tests delayed privatization for `targets ... private(..)` for adjustable arrays.
! Tests different allocation 

! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --enable-delayed-privatization-staging \
! RUN: -mmlir --enable-gpu-heap-alloc  -o - %s 2>&1 | FileCheck %s --check-prefix=CPU

! RUN: %if amdgpu-registered-target %{ \
! RUN:   %flang_fc1 -triple amdgcn-amd-amdhsa -emit-hlfir  \
! RUN:     -fopenmp -fopenmp-is-target-device \
! RUN:     -mmlir --enable-delayed-privatization-staging \
! RUN:     -mmlir --enable-gpu-heap-alloc \
! RUN:     -o - %s 2>&1 | \
! RUN:   FileCheck %s --check-prefix=GPU-HEAP \
! RUN: %}

! RUN: %if amdgpu-registered-target %{ \
! RUN:   %flang_fc1 -triple amdgcn-amd-amdhsa -emit-hlfir  \
! RUN:     -fopenmp -fopenmp-is-target-device \
! RUN:     -mmlir --enable-delayed-privatization-staging \
! RUN:     -o - %s 2>&1 | \
! RUN:   FileCheck %s --check-prefix=GPU-STACK  \
! RUN: %}

subroutine target_adjustable_array(n_size)
  implicit none
  integer, intent(in) :: n_size
  integer  :: alloc_var(n_size)

  !$omp target private(alloc_var)
    alloc_var = 1
  !$omp end target
end subroutine target_adjustable_array

! CPU-LABEL: omp.private {type = private}
! CPU-SAME:    @[[VAR_PRIVATIZER_SYM:.*]] : ![[DESC_TYPE:.*]] init {
! CPU-NEXT:  ^bb0(%[[PRIV_ARG:.*]]: ![[TYPE:.*]], %[[PRIV_ALLOC:.*]]: ![[TYPE]]):
! CPU-NEXT:  %[[PRIV_ARG_VAL:.*]] = fir.load %[[PRIV_ARG]] : ![[TYPE]]
! CPU-NEXT:  %[[C0:.*]] = arith.constant 0 : index 
! CPU-NEXT:  %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[PRIV_ARG_VAL]], %[[C0]] : (![[DESC_TYPE]], index) -> (index, index, index)
! CPU-NEXT:  %[[SHAPE:.*]] = fir.shape %[[BOX_DIMS]]#1 : (index) -> !fir.shape<1>
! CPU-NEXT:  %[[PRIVATE_MEM:.*]] = fir.allocmem !fir.array<?xi32>, %[[BOX_DIMS]]#1
! CPU-NEXT:  %4:2 = hlfir.declare %3(%2) {[[NAME_ATTR:.*]]} : (![[HEAP_ARRAY_TYPE:.*]], !fir.shape<1>) -> (![[DESC_TYPE]], ![[HEAP_ARRAY_TYPE]])
! CPU:      omp.yield(%[[PRIV_ALLOC]] : ![[TYPE]])
! CPU-NEXT: } dealloc {
! CPU-NEXT: ^bb0(%[[PRIV_ARG:.*]]: ![[TYPE]]):
! CPU-NEXT:  %[[PRIV_ARG_VAL1:.*]] = fir.load %[[PRIV_ARG]] : ![[TYPE]]
! CPU-NEXT:  %[[ALLOC_ADDR:.*]] = fir.box_addr %[[PRIV_ARG_VAL1]] : (![[DESC_TYPE]]) -> ![[REF_ARRAY_TYPE:.*]]
! CPU:       %[[CONV:.*]] = fir.convert %[[ALLOC_ADDR]] : (![[REF_ARRAY_TYPE]]) -> ![[HEAP_ARRAY_TYPE]]
! CPU-NEXT:  fir.freemem %[[CONV]] : ![[HEAP_ARRAY_TYPE]]
! CPU:      omp.yield
! CPU-NEXT: }

! GPU-HEAP-LABEL: omp.private {type = private}
! GPU-HEAP-SAME:    @[[VAR_PRIVATIZER_SYM:.*]] : ![[DESC_TYPE:.*]] init {
! GPU-HEAP-NEXT:  ^bb0(%[[PRIV_ARG:.*]]: ![[TYPE:.*]], %[[PRIV_ALLOC:.*]]: ![[TYPE]]):
! GPU-HEAP-NEXT:  %[[PRIV_ARG_VAL:.*]] = fir.load %[[PRIV_ARG]] : ![[TYPE]]
! GPU-HEAP-NEXT:  %[[C0:.*]] = arith.constant 0 : index 
! GPU-HEAP-NEXT:  %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[PRIV_ARG_VAL]], %[[C0]] : (![[DESC_TYPE]], index) -> (index, index, index)
! GPU-HEAP-NEXT:  %[[SHAPE:.*]] = fir.shape %[[BOX_DIMS]]#1 : (index) -> !fir.shape<1>
! GPU-HEAP-NEXT:  %[[PRIVATE_MEM:.*]] = fir.allocmem !fir.array<?xi32>, %[[BOX_DIMS]]#1
! GPU-HEAP-NEXT:  %4:2 = hlfir.declare %3(%2) {[[NAME_ATTR:.*]]} : (![[HEAP_ARRAY_TYPE:.*]], !fir.shape<1>) -> (![[DESC_TYPE]], ![[HEAP_ARRAY_TYPE]])
! GPU-HEAP:      omp.yield(%[[PRIV_ALLOC]] : ![[TYPE]])
! GPU-HEAP-NEXT: } dealloc {
! GPU-HEAP-NEXT: ^bb0(%[[PRIV_ARG:.*]]: ![[TYPE]]):
! GPU-HEAP-NEXT:  %[[PRIV_ARG_VAL1:.*]] = fir.load %[[PRIV_ARG]] : ![[TYPE]]
! GPU-HEAP-NEXT:  %[[ALLOC_ADDR:.*]] = fir.box_addr %[[PRIV_ARG_VAL1]] : (![[DESC_TYPE]]) -> ![[REF_ARRAY_TYPE:.*]]
! GPU-HEAP:       %[[CONV:.*]] = fir.convert %[[ALLOC_ADDR]] : (![[REF_ARRAY_TYPE]]) -> ![[HEAP_ARRAY_TYPE]]
! GPU-HEAP-NEXT:  fir.freemem %[[CONV]] : ![[HEAP_ARRAY_TYPE]]
! GPU-HEAP:      omp.yield
! GPU-HEAP-NEXT: }

! GPU-STACK-LABEL: omp.private {type = private}
! GPU-STACK-SAME:    @[[VAR_PRIVATIZER_SYM:.*]] : ![[DESC_TYPE:.*]] init {
! GPU-STACK-NEXT:  ^bb0(%[[PRIV_ARG:.*]]: ![[TYPE:.*]], %[[PRIV_ALLOC:.*]]: ![[TYPE]]):
! GPU-STACK-NEXT:  %[[PRIV_ARG_VAL:.*]] = fir.load %[[PRIV_ARG]] : ![[TYPE]]
! GPU-STACK-NEXT:  %[[C0:.*]] = arith.constant 0 : index 
! GPU-STACK-NEXT:  %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[PRIV_ARG_VAL]], %[[C0]] : (![[DESC_TYPE]], index) -> (index, index, index)
! GPU-STACK-NOT:   %[[PRIVATE_MEM:.*]] = fir.allocmem
! GPU-STACK:       %[[ALLOCA_ADDR:.*]] = fir.alloca !fir.array<?xi32>, %[[BOX_DIMS]]#1 {[[NAME_ATTR:.*]]}
! GPU-STACK:      omp.yield(%[[PRIV_ALLOC]] : ![[TYPE]])

