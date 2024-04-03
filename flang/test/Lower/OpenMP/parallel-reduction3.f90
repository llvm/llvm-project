! RUN: bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s


!CHECK: omp.declare_reduction @[[REDUNCTION_FUNC:.*]] : !fir.ref<!fir.box<!fir.array<?xi32>>> init {
!CHECK:  ^bb0(%{{.*}}: !fir.ref<!fir.box<!fir.array<?xi32>>>):
!CHECK:    %c0_i32 = arith.constant 0 : i32
!CHECK:    %0 = fir.load %arg0 : !fir.ref<!fir.box<!fir.array<?xi32>>>
!CHECK:    %c0 = arith.constant 0 : index
!CHECK:    %1:3 = fir.box_dims %0, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
!CHECK:    %2 = fir.shape %1#1 : (index) -> !fir.shape<1>
!CHECK:    %3 = fir.alloca !fir.array<?xi32>, %1#1 {bindc_name = ".tmp"}
!CHECK:    %[[ACC_DECL:.*]]:2 = hlfir.declare %3(%2) {uniq_name = ".tmp"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>)
!CHECK:    hlfir.assign %c0_i32 to %4#0 : i32, !fir.box<!fir.array<?xi32>>
!CHECK:    %[[ACC_REF:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
!CHECK:    fir.store %4#0 to %[[ACC_REF]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
!CHECK:    omp.yield(%[[ACC_REF]] : !fir.ref<!fir.box<!fir.array<?xi32>>>)
!CHECK:  } combiner {
!CHECK:  ^bb0(%arg0: !fir.ref<!fir.box<!fir.array<?xi32>>>, %arg1: !fir.ref<!fir.box<!fir.array<?xi32>>>):
!CHECK:    %0 = fir.load %arg0 : !fir.ref<!fir.box<!fir.array<?xi32>>>
!CHECK:    %1 = fir.load %arg1 : !fir.ref<!fir.box<!fir.array<?xi32>>>
!CHECK:    %c0 = arith.constant 0 : index
!CHECK:    %2:3 = fir.box_dims %0, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
!CHECK:    %3 = fir.shape_shift %2#0, %2#1 : (index, index) -> !fir.shapeshift<1>
!CHECK:    %c1 = arith.constant 1 : index
!CHECK:    fir.do_loop %arg2 = %c1 to %2#1 step %c1 unordered {
!CHECK:      %[[C0:.*]] = fir.array_coor %0(%3) %arg2 : (!fir.box<!fir.array<?xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
!CHECK:      %[[C1:.*]] = fir.array_coor %1(%3) %arg2 : (!fir.box<!fir.array<?xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
!CHECK:      %[[C0_REF:.*]] = fir.load %[[C0]] : !fir.ref<i32>
!CHECK:      %[[C1_REF:.*]] = fir.load %[[C1]] : !fir.ref<i32>
!CHECK:      %8 = arith.addi %[[C0_REF]], %[[C1_REF]] : i32
!CHECK:      fir.store %8 to %[[ACC_DECL]] : !fir.ref<i32>
!CHECK:    }
!CHECK:    omp.yield(%arg0 : !fir.ref<!fir.box<!fir.array<?xi32>>>)
!CHECK:  } 
!CHECK:  func.func @_QPs(%arg0: !fir.ref<i32> {fir.bindc_name = "x"}) {
!CHECK:    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFsEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %1 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsEi"}
!CHECK:    %2:2 = hlfir.declare %1 {uniq_name = "_QFsEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %8 = fir.alloca !fir.array<?xi32>, %7 {bindc_name = "c", uniq_name = "_QFsEc"}
!CHECK:    %9 = fir.shape %7 : (index) -> !fir.shape<1>
!CHECK:    %10:2 = hlfir.declare %8(%9) {uniq_name = "_QFsEc"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>)
!CHECK:    hlfir.assign %c0_i32 to %10#0 : i32, !fir.box<!fir.array<?xi32>>
!CHECK:    omp.parallel {
!CHECK:      %15 = fir.alloca i32 {adapt.valuebyref, pinned}
!CHECK:      %16:2 = hlfir.declare %15 {uniq_name = "_QFsEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:      %[[LB:.*]] = arith.constant 1 : i32
!CHECK:      %[[UB:.*]] = arith.constant 100 : i32
!CHECK:      %[[STEP:.*]] = arith.constant 1 : i32
!CHECK:      %17 = fir.alloca !fir.box<!fir.array<?xi32>>
!CHECK:      fir.store %10#0 to %17 : !fir.ref<!fir.box<!fir.array<?xi32>>>
!CHECK:      omp.wsloop byref reduction(@[[REDUNCTION_FUNC]] %17 -> %arg2 : !fir.ref<!fir.box<!fir.array<?xi32>>>)  for  (%arg1) : i32 = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]]) {
!CHECK:        omp.yield
!CHECK:      }
!CHECK:      omp.terminator
!CHECK:    }

subroutine s(x)
    integer :: x
    integer :: c(x)
    c = 0
    !$omp parallel do reduction(+:c)
    do i = 1, 100
        c = c + i
    end do
    !$omp end parallel do

    if (c(1) /= 5050) stop 1
end subroutine s