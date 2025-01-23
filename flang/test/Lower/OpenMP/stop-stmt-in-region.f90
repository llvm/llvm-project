! This test checks lowering of stop statement in OpenMP region.

! RUN: %flang_fc1 -flang-experimental-hlfir -emit-hlfir -fopenmp %s -o - | FileCheck %s
! RUN: bbc -hlfir -emit-hlfir -fopenmp %s -o - | FileCheck %s


! CHECK-LABEL: func.func @_QPtest_stop_in_region1() {
! CHECK:         omp.parallel   {
! CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_1:.*]] = arith.constant false
! CHECK:           %[[VAL_2:.*]] = arith.constant false
! CHECK:           fir.call @_FortranAStopStatement(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) {{.*}} : (i32, i1, i1) -> ()
! CHECK-NOT:       fir.unreachable
! CHECK:           omp.terminator
! CHECK:         }
! CHECK:         return
! CHECK:       }

subroutine test_stop_in_region1()
  !$omp parallel
    stop 1
  !$omp end parallel
end

! CHECK-LABEL: func.func @_QPtest_stop_in_region2() {
! CHECK:         %{{.*}} = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFtest_stop_in_region2Ex"}
! CHECK:         omp.parallel   {
! CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_2:.*]] = arith.constant false
! CHECK:           %[[VAL_3:.*]] = arith.constant false
! CHECK:           fir.call @_FortranAStopStatement(%[[VAL_1]], %[[VAL_2]], %[[VAL_3]]) {{.*}} : (i32, i1, i1) -> ()
! CHECK:           omp.terminator
! CHECK:         }
! CHECK:         return
! CHECK:       }

subroutine test_stop_in_region2()
  integer :: x
  !$omp parallel
    stop 1
    x = 2
  !$omp end parallel
end

! CHECK-LABEL: func.func @_QPtest_stop_in_region3() {
! CHECK:         %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFtest_stop_in_region3Ex"}
! CHECK:         %[[VAL_0_DECL:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest_stop_in_region3Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:         omp.parallel   {
! CHECK:           %[[VAL_1:.*]] = arith.constant 3 : i32
! CHECK:           hlfir.assign %[[VAL_1]] to %[[VAL_0_DECL]]#0 : i32, !fir.ref<i32>
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_0_DECL]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_4:.*]] = arith.cmpi sgt, %[[VAL_2]], %[[VAL_3]] : i32
! CHECK:           cf.cond_br %[[VAL_4]], ^bb1, ^bb2
! CHECK:         ^bb1:
! CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_0_DECL]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_6:.*]] = arith.constant false
! CHECK:           %[[VAL_7:.*]] = arith.constant false
! CHECK:           fir.call @_FortranAStopStatement(%[[VAL_5]], %[[VAL_6]], %[[VAL_7]]) {{.*}} : (i32, i1, i1) -> ()
! CHECK:           omp.terminator
! CHECK:         ^bb2:
! CHECK:           omp.terminator
! CHECK:         }
! CHECK:         return
! CHECK:       }

subroutine test_stop_in_region3()
  integer :: x
  !$omp parallel
    x = 3
    if (x > 1) stop x
  !$omp end parallel
end

! CHECK-LABEL: func.func @_QPtest_stop_in_region4() {

! CHECK:         %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_stop_in_region4Ei"}
! CHECK:         %[[VAL_1_DECL:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFtest_stop_in_region4Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFtest_stop_in_region4Ex"}
! CHECK:         %[[VAL_2_DECL:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFtest_stop_in_region4Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

! CHECK:         %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "i", pinned, {{.*}}}
! CHECK:         %[[VAL_0_DECL:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest_stop_in_region4Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

! CHECK:         %[[VAL_3:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_4:.*]] = arith.constant 10 : i32
! CHECK:         %[[VAL_5:.*]] = arith.constant 1 : i32
! CHECK:         omp.wsloop {
! CHECK-NEXT:      omp.loop_nest (%[[VAL_6:.*]]) : i32 = (%[[VAL_3]]) to (%[[VAL_4]]) inclusive step (%[[VAL_5]]) {
! CHECK:             fir.store %[[VAL_6]] to %[[VAL_0_DECL]]#1 : !fir.ref<i32>
! CHECK:             cf.br ^bb1
! CHECK:           ^bb1:
! CHECK:             %[[VAL_7:.*]] = arith.constant 3 : i32
! CHECK:             hlfir.assign %[[VAL_7]] to %[[VAL_2_DECL]]#0 : i32, !fir.ref<i32>
! CHECK:             %[[VAL_8:.*]] = fir.load %[[VAL_2_DECL]]#0 : !fir.ref<i32>
! CHECK:             %[[VAL_9:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_10:.*]] = arith.cmpi sgt, %[[VAL_8]], %[[VAL_9]] : i32
! CHECK:             cf.cond_br %[[VAL_10]], ^bb2, ^bb3
! CHECK:           ^bb2:
! CHECK:             %[[VAL_11:.*]] = fir.load %[[VAL_2_DECL]]#0 : !fir.ref<i32>
! CHECK:             %[[VAL_12:.*]] = arith.constant false
! CHECK:             %[[VAL_13:.*]] = arith.constant false
! CHECK:             fir.call @_FortranAStopStatement(%[[VAL_11]], %[[VAL_12]], %[[VAL_13]]) {{.*}} : (i32, i1, i1) -> ()
! CHECK:             omp.yield
! CHECK:           ^bb3:
! CHECK:             omp.yield
! CHECK:           }
! CHECK:         }
! CHECK:         cf.br ^bb1
! CHECK:       ^bb1:
! CHECK:         return
! CHECK:       }

subroutine test_stop_in_region4()
  integer :: x
  !$omp do
  do i = 1, 10
    x = 3
    if (x > 1) stop x
  enddo
  !$omp end do
end


!CHECK-LABEL: func.func @_QPtest_stop_in_region5
!CHECK:   omp.parallel   {
!CHECK:     {{.*}} fir.call @_FortranAStopStatement({{.*}}, {{.*}}, {{.*}}) fastmath<contract> : (i32, i1, i1) -> ()
!CHECK:     omp.terminator
!CHECK:   }
!CHECK:   return

subroutine test_stop_in_region5()
  !$omp parallel
  block
    stop 1
  end block
  !$omp end parallel
end

!CHECK-LABEL: func.func @_QPtest_stop_in_region6
!CHECK:  omp.parallel   {
!CHECK:    cf.cond_br %{{.*}}, ^[[BB1:.*]], ^[[BB2:.*]]
!CHECK:  ^[[BB1]]:
!CHECK:    {{.*}}fir.call @_FortranAStopStatement({{.*}}, {{.*}}, {{.*}}) fastmath<contract> : (i32, i1, i1) -> ()
!CHECK:    omp.terminator
!CHECK:  ^[[BB2]]:
!CHECK:    {{.*}}fir.call @_FortranAStopStatement({{.*}}, {{.*}}, {{.*}}) fastmath<contract> : (i32, i1, i1) -> ()
!CHECK:    omp.terminator
!CHECK:  }
!CHECK:  return

subroutine test_stop_in_region6(x)
  integer :: x
  !$omp parallel
  if (x .gt. 1) then
    stop 1
  else
    stop 2
  end if
  !$omp end parallel
end
