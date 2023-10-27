! This test checks lowering of OpenMP Threadprivate Directive.
! Test for threadprivate variable double use in use association.

!RUN: %flang_fc1 -emit-hlfir -flang-experimental-hlfir -fopenmp %s -o - | FileCheck %s
!RUN: bbc -emit-hlfir -fopenmp %s -o - | FileCheck %s

! CHECK-LABEL: fir.global @_QMmEx : i32
module m
  integer :: x
  !$omp threadprivate(x)
end

! CHECK-LABEL: func.func @_QMm2Ptest() {
! CHECK:         %[[VAL_0:.*]] = fir.address_of(@_QMmEx) : !fir.ref<i32>
! CHECK:         %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QMmEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:         %[[VAL_2:.*]] = omp.threadprivate %[[VAL_1]]#1 : !fir.ref<i32> -> !fir.ref<i32>
! CHECK:         %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QMmEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:         fir.call @_QPbar(%[[VAL_3]]#1) {{.*}}: (!fir.ref<i32>) -> ()
! CHECK:         return
! CHECK:       }

! CHECK-LABEL: func.func @_QMm2FtestPinternal_test() {
! CHECK:         %[[VAL_0:.*]] = fir.address_of(@_QMmEx) : !fir.ref<i32>
! CHECK:         %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QMmEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:         %[[VAL_2:.*]] = omp.threadprivate %[[VAL_1]]#1 : !fir.ref<i32> -> !fir.ref<i32>
! CHECK:         %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QMmEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:         fir.call @_QPbar(%[[VAL_3]]#1) {{.*}}: (!fir.ref<i32>) -> ()
! CHECK:         return
! CHECK:       }

module m2
  use m
 contains
  subroutine test()
    use m
    call bar(x)
   contains
    subroutine internal_test()
      use m
      call bar(x)
    end
  end
end
