! Test that function calls to declare simd functions inside !$omp simd loops
! get the omp.simd_inline_boost attribute when -openmp-simd-inline-boost is set.
!RUN: %flang_fc1 -emit-mlir -fopenmp -mmlir -openmp-simd-inline-boost %s -o - | FileCheck %s

real function foo(v)
  !$omp declare simd
  real, intent(in) :: v
  foo = v * v
end function

! CHECK-LABEL: func.func @_QPtest_simd
subroutine test_simd(x, n)
  implicit none
  integer, intent(in) :: n
  real, intent(inout) :: x(n)
  integer :: i
  interface
    real function foo(v)
      !$omp declare simd
      real, intent(in) :: v
    end function
  end interface
  !$omp simd
  do i = 1, n
    ! CHECK: fir.call @_QPfoo({{.*}}) {{.*}}omp.simd_inline_boost
    x(i) = foo(x(i))
  end do
  !$omp end simd
end subroutine

! Calls to declare simd functions outside !$omp simd should NOT get the attribute.
! CHECK-LABEL: func.func @_QPno_simd
subroutine no_simd(x)
  implicit none
  real, intent(inout) :: x
  interface
    real function foo(v)
      !$omp declare simd
      real, intent(in) :: v
    end function
  end interface
  ! CHECK: fir.call @_QPfoo({{.*}})
  ! CHECK-NOT: omp.simd_inline_boost
  x = foo(x)
end subroutine
