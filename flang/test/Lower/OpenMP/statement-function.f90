! Test privatization within OpenMP constructs containing statement functions.
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

!CHECK-LABEL: func @_QPtest_implicit_use
!CHECK:         %[[IEXP:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_implicit_useEiexp"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:         %[[IIMP:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_implicit_useEiimp"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:         omp.parallel private({{.*firstprivate.*}} %[[IEXP]]#0 -> %[[PRIV_IEXP:[^,]+]],
!CHECK-SAME:                         {{.*firstprivate.*}} %[[IIMP]]#0 -> %[[PRIV_IIMP:.*]] : !fir.ref<i32>, !fir.ref<i32>)
!CHECK:           %{{.*}}:2 = hlfir.declare %[[PRIV_IEXP]] {uniq_name = "_QFtest_implicit_useEiexp"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:           %{{.*}}:2 = hlfir.declare %[[PRIV_IIMP]] {uniq_name = "_QFtest_implicit_useEiimp"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
subroutine test_implicit_use()
  implicit none
  integer :: iexp, iimp
  integer, external :: ifun
  integer :: sf

  sf(iexp)=ifun(iimp)+iexp
  !$omp parallel default(firstprivate)
      iexp = sf(iexp)
  !$omp end parallel
end subroutine

!CHECK-LABEL: func @_QPtest_implicit_use2
!CHECK:         %[[IEXP:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_implicit_use2Eiexp"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:         %[[IIMP:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_implicit_use2Eiimp"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:         omp.task
!CHECK:           %[[PRIV_IEXP:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_implicit_use2Eiexp"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:           %[[TEMP0:.*]] = fir.load %[[IEXP]]#0 : !fir.ref<i32>
!CHECK:           hlfir.assign %[[TEMP0]] to %[[PRIV_IEXP]]#0 : i32, !fir.ref<i32>
!CHECK:           %[[PRIV_IIMP:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_implicit_use2Eiimp"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:           %[[TEMP1:.*]] = fir.load %[[IIMP]]#0 : !fir.ref<i32>
!CHECK:           hlfir.assign %[[TEMP1]] to %[[PRIV_IIMP]]#0 : i32, !fir.ref<i32>
subroutine test_implicit_use2()
  implicit none
  integer :: iexp, iimp
  integer, external :: ifun
  integer :: sf

  sf(iexp)=ifun(iimp)
  !$omp task
      iexp = sf(iexp)
  !$omp end task
end subroutine
