!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!RUN: %flang_fc1 -emit-llvm -fopenmp %s -o - | FileCheck %s --check-prefixes HOST,ALL
!RUN: %flang_fc1 -emit-llvm -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s --check-prefixes DEVICE,ALL

!HOST: define {{.*}}@{{.*}}before{{.*}}(
!DEVICE-NOT: define {{.*}}@before{{.*}}(
!DEVICE-NOT: declare {{.*}}@before{{.*}}
integer function before(x)
   integer, intent(in) :: x
   before = x + 200
end function

!ALL: define {{.*}}@{{.*}}main{{.*}}(
program main
   integer :: x, before, after
   !$omp target map(tofrom : x)
      x = 100
   !$omp end target
   !HOST: call {{.*}}@{{.*}}before{{.*}}(
   !DEVICE-NOT: call {{.*}}@before{{.*}}(
   !HOST: call {{.*}}@{{.*}}after{{.*}}(
   !DEVICE-NOT: call {{.*}}@after{{.*}}(
   x = x + before(x) + after(x)
end program

!HOST: define {{.*}}@{{.*}}after{{.*}}(
!DEVICE-NOT: define {{.*}}@after{{.*}}(
!DEVICE-NOT: declare {{.*}}@after{{.*}}
integer function after(x)
   integer, intent(in) :: x
   after = x + 300
end function

!ALL: define {{.*}}@{{.*}}before_target{{.*}}(
subroutine before_target(x)
   integer, intent(out) :: x
   !$omp target map(from: x)
      x = 1
   !$omp end target
end subroutine

!ALL: define {{.*}}@{{.*}}middle{{.*}}(
subroutine middle()
   integer :: x
   !$omp target map(from: x)
      x = 0
   !$omp end target
   !HOST: call {{.*}}@{{.*}}before_target{{.*}}(
   !DEVICE-NOT: call {{.*}}@{{.*}}before_target{{.*}}(
   !HOST: call {{.*}}@{{.*}}after_target{{.*}}(
   !DEVICE-NOT: call {{.*}}@{{.*}}after_target{{.*}}(
   call before_target(x)
   call after_target(x)
end subroutine

!ALL: define {{.*}}@{{.*}}after_target{{.*}}(
subroutine after_target(x)
   integer, intent(out) :: x
   !$omp target map(from:x)
      x = 2
   !$omp end target
end subroutine
