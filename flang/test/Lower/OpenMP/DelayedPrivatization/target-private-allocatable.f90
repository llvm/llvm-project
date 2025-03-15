! Tests delayed privatization for `targets ... private(..)` for allocatables.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --openmp-enable-delayed-privatization-staging \
! RUN:   -o - %s 2>&1 | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp --openmp-enable-delayed-privatization-staging -o - %s 2>&1 \
! RUN:   | FileCheck %s

subroutine target_allocatable
  implicit none
  integer, allocatable :: alloc_var

  !$omp target private(alloc_var)
    alloc_var = 10
  !$omp end target
end subroutine target_allocatable

! CHECK-LABEL: omp.private {type = private}
! CHECK-SAME:    @[[VAR_PRIVATIZER_SYM:.*]] :
! CHECK-SAME:      [[DESC_TYPE:!fir.box<!fir.heap<i32>>]] init {
! CHECK:  ^bb0(%[[PRIV_ARG:.*]]: [[TYPE:!fir.ref<!fir.box<!fir.heap<i32>>>]], %[[PRIV_ALLOC:.*]]: [[TYPE]]):

! CHECK-NEXT:   %[[PRIV_ARG_VAL:.*]] = fir.load %[[PRIV_ARG]] : [[TYPE]]
! CHECK-NEXT:   %[[PRIV_ARG_BOX:.*]] = fir.box_addr %[[PRIV_ARG_VAL]] : ([[DESC_TYPE]]) -> !fir.heap<i32>
! CHECK-NEXT:   %[[PRIV_ARG_ADDR:.*]] = fir.convert %[[PRIV_ARG_BOX]] : (!fir.heap<i32>) -> i64
! CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : i64
! CHECK-NEXT:   %[[ALLOC_COND:.*]] = arith.cmpi eq, %[[PRIV_ARG_ADDR]], %[[C0]] : i64

! CHECK-NEXT:   fir.if %[[ALLOC_COND]] {
! CHECK-NEXT:     %[[ZERO_BOX:.*]] = fir.embox %[[PRIV_ARG_BOX]] : (!fir.heap<i32>) -> [[DESC_TYPE]]
! CHECK-NEXT:     fir.store %[[ZERO_BOX]] to %[[PRIV_ALLOC]] : [[TYPE]]
! CHECK-NEXT:   } else {
! CHECK-NEXT:     %[[PRIV_ALLOCMEM:.*]] = fir.allocmem i32
! CHECK-NEXT:     %[[PRIV_ALLOCMEM_BOX:.*]] = fir.embox %[[PRIV_ALLOCMEM]] : (!fir.heap<i32>) -> [[DESC_TYPE]]
! CHECK-NEXT:     fir.store %[[PRIV_ALLOCMEM_BOX]] to %[[PRIV_ALLOC]] : [[TYPE]]
! CHECK-NEXT:   }

! CHECK-NEXT:   omp.yield(%[[PRIV_ALLOC]] : [[TYPE]])

! CHECK-NEXT: } dealloc {
! CHECK-NEXT: ^bb0(%[[PRIV_ARG:.*]]: [[TYPE]]):

! CHECK-NEXT:   %[[PRIV_VAL:.*]] = fir.load %[[PRIV_ARG]]
! CHECK-NEXT:   %[[PRIV_ADDR:.*]] = fir.box_addr %[[PRIV_VAL]]
! CHECK-NEXT:   %[[PRIV_ADDR_I64:.*]] = fir.convert %[[PRIV_ADDR]]
! CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : i64
! CHECK-NEXT:   %[[PRIV_NULL_COND:.*]] = arith.cmpi ne, %[[PRIV_ADDR_I64]], %[[C0]] : i64

! CHECK-NEXT:   fir.if %[[PRIV_NULL_COND]] {
! CHECK-NEXT:     fir.freemem %[[PRIV_ADDR]]
! CHECK-NEXT:   }

! CHECK-NEXT:   omp.yield
! CHECK-NEXT: }


! CHECK-LABEL: func.func @_QPtarget_allocatable() {

! CHECK:  %[[VAR_ALLOC:.*]] = fir.alloca [[DESC_TYPE]]
! CHECK-SAME: {bindc_name = "alloc_var", {{.*}}}
! CHECK:  %[[VAR_DECL:.*]]:2 = hlfir.declare %[[VAR_ALLOC]]

! CHECK: %[[MAP_VAR:.*]] = omp.map.info var_ptr(%[[VAR_DECL]]#0 : [[TYPE]], [[DESC_TYPE]])
! CHECK-SAME: map_clauses(to) capture(ByRef) -> [[TYPE]]
! CHECK:  omp.target map_entries(%[[MAP_VAR]] -> %arg0 : [[TYPE]]) private(
! CHECK-SAME: @[[VAR_PRIVATIZER_SYM]] %[[VAR_DECL]]#0 -> %{{.*}} : [[TYPE]]) {
