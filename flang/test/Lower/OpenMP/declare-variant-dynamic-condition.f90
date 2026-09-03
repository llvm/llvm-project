! RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s

! A run-time (non-constant) USER condition in a DECLARE VARIANT match clause is
! resolved at the call site as an if/else cascade: the variant is called when
! the condition holds, otherwise the base.

! CHECK-LABEL: func.func @_QPtest_single
! CHECK: %[[COND:.*]] = arith.cmpi sgt
! CHECK: fir.if %[[COND]] {
! CHECK:   fir.call @_QFtest_singlePvsub
! CHECK: } else {
! CHECK:   fir.call @_QFtest_singlePbase
! CHECK: }
subroutine test_single(x)
  integer :: x
  call base()
contains
  subroutine base
    !$omp declare variant (base:vsub) match (user={condition(x > 0)})
  end subroutine base
  subroutine vsub
  end subroutine vsub
end subroutine test_single

! Two run-time conditions form a ranked if/else-if cascade, with the base as the
! final fallback.

! CHECK-LABEL: func.func @_QPtest_chain
! CHECK: fir.if
! CHECK:   fir.call @_QFtest_chainPv1
! CHECK: } else {
! CHECK:   fir.if
! CHECK:     fir.call @_QFtest_chainPv2
! CHECK:   } else {
! CHECK:     fir.call @_QFtest_chainPbase
subroutine test_chain(a, b)
  logical :: a, b
  call base()
contains
  subroutine base
    !$omp declare variant (base:v1) match (user={condition(a)})
    !$omp declare variant (base:v2) match (user={condition(b)})
  end subroutine base
  subroutine v1
  end subroutine v1
  subroutine v2
  end subroutine v2
end subroutine test_chain

! A variant with both a static selector and a run-time condition: the static
! part decides applicability at compile time (only inside the parallel region),
! and the condition guards the call at run time.

! CHECK-LABEL: func.func @_QPtest_mix
! CHECK: fir.call @_QFtest_mixPbase
! CHECK: omp.parallel {
! CHECK:   fir.if
! CHECK:     fir.call @_QFtest_mixPvsub
! CHECK:   } else {
! CHECK:     fir.call @_QFtest_mixPbase
subroutine test_mix(x)
  integer :: x
  call base()
  !$omp parallel
  call base()
  !$omp end parallel
contains
  subroutine base
    !$omp declare variant (base:vsub) match (construct={parallel}, user={condition(x > 0)})
  end subroutine base
  subroutine vsub
  end subroutine vsub
end subroutine test_mix
