!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s --check-prefix=HOST
!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s --check-prefix=DEVICE

! Test that a bare '!$omp declare target' inside an interface body that
! appears in a *named* main program does not incorrectly mark the main
! program (_QQmain) as a declare-target function while still correctly
! marking the declared subroutine (sub_a) as device_type(nohost).
!
! In host compilation _QQmain must not be tagged with declare_target at all.
! In device compilation the MarkDeclareTargetPass may annotate _QQmain with
! device_type(host) (harmless); the bug was device_type(any) which caused
! _QQmain to be emitted into the device image.

! HOST-NOT: func.func @_QQmain{{.*}}omp.declare_target
! DEVICE-NOT: func.func @_QQmain{{.*}}device_type = (any)
! DEVICE-NOT: func.func @_QQmain{{.*}}device_type = (nohost)
! HOST: func.func private @_QPsub_a{{.*}}device_type = (nohost)
! DEVICE: func.func private @_QPsub_a{{.*}}device_type = (nohost)

program named_main
  interface
    subroutine sub_a(x)
      implicit none
      !$omp declare target
      integer, intent(inout) :: x
    end subroutine sub_a
  end interface
  integer :: v = 0
  !$omp target
    call sub_a(v)
  !$omp end target
end program named_main
