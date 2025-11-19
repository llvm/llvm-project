! Tests delayed privatization for `targets ... private(..)` for allocatables.
! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --enable-delayed-privatization-staging \
! RUN:   -o - %s 2>&1 | FileCheck %s --check-prefix=CPU

! RUN: bbc -emit-hlfir -fopenmp --enable-delayed-privatization-staging -o - %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=CPU

! RUN: %if amdgpu-registered-target %{ \
! RUN:   %flang_fc1 -triple amdgcn-amd-amdhsa -emit-hlfir  \
! RUN:     -fopenmp -fopenmp-is-target-device \
! RUN:     -mmlir --enable-delayed-privatization-staging \
! RUN:     -o - %s 2>&1 | \
! RUN:   FileCheck %s --check-prefix=GPU  \
! RUN: %}

! RUN: bbc -emit-hlfir -fopenmp --enable-delayed-privatization-staging \
! RUN:    -fopenmp-is-target-device -fopenmp-is-gpu -o - %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=GPU

subroutine target_allocatable
  implicit none
  integer, allocatable :: alloc_var

  !$omp target private(alloc_var)
    alloc_var = 10
  !$omp end target
end subroutine target_allocatable

! CPU-LABEL: omp.private {type = private}
! CPU-SAME:    @[[VAR_PRIVATIZER_SYM:.*]] :
! CPU-SAME:      [[DESC_TYPE:!fir.box<!fir.heap<i32>>]] init {
! CPU:  ^bb0(%[[PRIV_ARG:.*]]: [[TYPE:!fir.ref<!fir.box<!fir.heap<i32>>>]], %[[PRIV_ALLOC:.*]]: [[TYPE]]):

! CPU-NEXT:   %[[PRIV_ARG_VAL:.*]] = fir.load %[[PRIV_ARG]] : [[TYPE]]
! CPU-NEXT:   %[[PRIV_ARG_BOX:.*]] = fir.box_addr %[[PRIV_ARG_VAL]] : ([[DESC_TYPE]]) -> !fir.heap<i32>
! CPU-NEXT:   %[[PRIV_ARG_ADDR:.*]] = fir.convert %[[PRIV_ARG_BOX]] : (!fir.heap<i32>) -> i64
! CPU-NEXT:   %[[C0:.*]] = arith.constant 0 : i64
! CPU-NEXT:   %[[ALLOC_COND:.*]] = arith.cmpi eq, %[[PRIV_ARG_ADDR]], %[[C0]] : i64

! CPU-NEXT:   fir.if %[[ALLOC_COND]] {
! CPU-NEXT:     %[[ZERO_BOX:.*]] = fir.embox %[[PRIV_ARG_BOX]] : (!fir.heap<i32>) -> [[DESC_TYPE]]
! CPU-NEXT:     fir.store %[[ZERO_BOX]] to %[[PRIV_ALLOC]] : [[TYPE]]
! CPU-NEXT:   } else {
! CPU-NEXT:     %[[PRIV_ALLOCMEM:.*]] = fir.allocmem i32
! CPU-NEXT:     %[[PRIV_ALLOCMEM_BOX:.*]] = fir.embox %[[PRIV_ALLOCMEM]] : (!fir.heap<i32>) -> [[DESC_TYPE]]
! CPU-NEXT:     fir.store %[[PRIV_ALLOCMEM_BOX]] to %[[PRIV_ALLOC]] : [[TYPE]]
! CPU-NEXT:   }

! CPU-NEXT:   omp.yield(%[[PRIV_ALLOC]] : [[TYPE]])

! CPU-NEXT: } dealloc {
! CPU-NEXT: ^bb0(%[[PRIV_ARG:.*]]: [[TYPE]]):

! CPU-NEXT:   %[[PRIV_VAL:.*]] = fir.load %[[PRIV_ARG]]
! CPU-NEXT:   %[[PRIV_ADDR:.*]] = fir.box_addr %[[PRIV_VAL]]
! CPU-NEXT:   %[[PRIV_ADDR_I64:.*]] = fir.convert %[[PRIV_ADDR]]
! CPU-NEXT:   %[[C0:.*]] = arith.constant 0 : i64
! CPU-NEXT:   %[[PRIV_NULL_COND:.*]] = arith.cmpi ne, %[[PRIV_ADDR_I64]], %[[C0]] : i64

! CPU-NEXT:   fir.if %[[PRIV_NULL_COND]] {
! CPU-NEXT:     fir.freemem %[[PRIV_ADDR]]
! CPU-NEXT:   }

! CPU-NEXT:   omp.yield
! CPU-NEXT: }


! CPU-LABEL: func.func @_QPtarget_allocatable() {

! CPU:  %[[VAR_ALLOC:.*]] = fir.alloca [[DESC_TYPE]]
! CPU-SAME: {bindc_name = "alloc_var", {{.*}}}
! CPU:  %[[VAR_DECL:.*]]:2 = hlfir.declare %[[VAR_ALLOC]]
! CPU:  %[[BASE_ADDR:.*]] = fir.box_offset %[[VAR_DECL]]#0 base_addr : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> [[MEMBER_TYPE:.*]]
! CPU:  %[[MEMBER:.*]] = omp.map.info var_ptr(%[[VAR_DECL]]#0 : [[TYPE]], i32) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%[[BASE_ADDR]] : [[MEMBER_TYPE:.*]]) -> {{.*}}
! CPU:  %[[MAP_VAR:.*]] = omp.map.info var_ptr(%[[VAR_DECL]]#0 : [[TYPE]], [[DESC_TYPE]]) map_clauses({{.*}}to{{.*}}) capture(ByRef) members(%[[MEMBER]] : [0] : !fir.llvm_ptr<!fir.ref<i32>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>

! CPU:  omp.target map_entries(%[[MAP_VAR]] -> %arg0, %[[MEMBER]] -> %arg1 : [[TYPE]], [[MEMBER_TYPE]]) private(
! CPU-SAME: @[[VAR_PRIVATIZER_SYM]] %[[VAR_DECL]]#0 -> %{{.*}} [map_idx=0] : [[TYPE]]) {

! GPU-LABEL: omp.private {type = private} {{.*}} init {
! GPU:         fir.if %{{.*}} {
! GPU-NEXT:    %[[ZERO_BOX:.*]] = fir.embox %{{.*}}
! GPU-NEXT:     fir.store %[[ZERO_BOX]] to %{{.*}}
! GPU-NEXT:   } else {
! GPU-NOT:      fir.allocmem i32
! GPU-NEXT:     %[[PRIV_ALLOC:.*]] = fir.alloca i32
! GPU-NEXT:     %[[PRIV_ALLOC_BOX:.*]] = fir.embox %[[PRIV_ALLOC]]
! GPU-NEXT:     fir.store %[[PRIV_ALLOC_BOX]] to %{{.*}}
! GPU-NEXT:   }
! GPU-NEXT:   omp.yield(%{{.*}})
