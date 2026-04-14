! Test that function calls inside !$omp simd loops get boosted inline thresholds.
!RUN: %flang_fc1 -emit-llvm -fopenmp %s -o - | FileCheck %s

! CHECK-LABEL: define {{.*}} @test_simd_
subroutine test_simd(x, n)
  implicit none
  integer, intent(in) :: n
  real, intent(inout) :: x(n)
  integer :: i
  interface
    real function foo(v)
      real, intent(in) :: v
    end function
  end interface
  !$omp simd
  do i = 1, n
    ! CHECK: call {{.*}}@foo_({{.*}}) #[[BOOST:[0-9]+]]
    x(i) = foo(x(i))
  end do
  !$omp end simd
end subroutine

! Calls outside !$omp simd should NOT get the attribute.
! CHECK-LABEL: define {{.*}} @no_simd_
subroutine no_simd(x)
  implicit none
  real, intent(inout) :: x
  interface
    real function foo(v)
      real, intent(in) :: v
    end function
  end interface
  ! CHECK: call {{.*}}@foo_({{.*}})
  ! CHECK-NOT: call {{.*}}@foo_({{.*}}) #[[BOOST]]
  x = foo(x)
end subroutine

! CHECK: attributes #[[BOOST]] = {{{.*}}"function-inline-threshold-bonus"="2000"{{.*}}}
