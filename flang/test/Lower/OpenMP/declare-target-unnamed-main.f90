!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s
!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-is-device %s -o - | FileCheck %s

! This test is a reduced version of the example in issue 63362.
! It aims to test that no crash occurs when declare target is
! utilised within an unnamed main program and that we still
! appropriately mark the function as declare target, even when
! unused within the target region.

!CHECK: func.func @_QPfoo(%{{.*}}: !fir.ref<f32>{{.*}}) -> f32 attributes {{{.*}}omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to), automap = false>{{.*}}}

interface
real function foo (x)
  !$omp declare target
  real, intent(in) :: x
end function foo
end interface
integer, parameter :: n = 1000
integer, parameter :: c = 100
integer :: i, j
real :: a(n)
do i = 1, n
a(i) = i
end do
do i = 1, n, c
  !$omp target map(a(i:i+c-1))
    !$omp parallel do
      do j = i, i + c - 1
        a(j) = a(j)
      end do
  !$omp end target
end do
do i = 1, n
if (a(i) /= i + 1) stop 1
end do
end
real function foo (x)
!$omp declare target
real, intent(in) :: x
foo = x + 1
end function foo
