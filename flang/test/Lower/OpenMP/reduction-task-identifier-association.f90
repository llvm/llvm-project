! RUN: %flang_fc1 -fopenmp -fopenmp-version=50 -emit-hlfir -o - %s | FileCheck %s

! CHECK-LABEL: omp.declare_reduction @_QQFlocal_ownermyred_i32
! CHECK: arith.constant 10 : i32
! CHECK-LABEL: omp.declare_reduction @_QQMred_defsmyred_i32
! CHECK: arith.constant 100 : i32

module red_defs
  implicit none
  !$omp declare reduction(myred : integer : omp_out = omp_out + 100 * omp_in) &
  !$omp& initializer(omp_priv = 0)
contains
  ! CHECK-LABEL: func.func @_QMred_defsPhost_owner(
  ! CHECK: omp.taskgroup task_reduction(@_QQMred_defsmyred_i32
  subroutine host_owner(x)
    integer :: x
    !$omp taskgroup task_reduction(myred: x)
    !$omp end taskgroup
  end subroutine

  ! CHECK-LABEL: func.func @_QMred_defsPhost_participant(
  ! CHECK: omp.parallel reduction(task @_QQMred_defsmyred_i32
  ! CHECK: omp.task in_reduction(@_QQMred_defsmyred_i32
  subroutine host_participant(x)
    integer :: x
    !$omp parallel reduction(task, myred: x)
    !$omp task in_reduction(myred: x)
    x = x + 1
    !$omp end task
    !$omp end parallel
  end subroutine
end module

! CHECK-LABEL: func.func @_QPuse_owner(
! CHECK: omp.taskgroup task_reduction(@_QQMred_defsmyred_i32
subroutine use_owner(x)
  use red_defs
  integer :: x
  !$omp taskgroup task_reduction(myred: x)
  !$omp end taskgroup
end subroutine

! CHECK-LABEL: func.func @_QPuse_participant(
! CHECK: omp.parallel reduction(task @_QQMred_defsmyred_i32
! CHECK: omp.task in_reduction(@_QQMred_defsmyred_i32
subroutine use_participant(x)
  use red_defs
  integer :: x
  !$omp parallel reduction(task, myred: x)
  !$omp task in_reduction(myred: x)
  x = x + 1
  !$omp end task
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPlocal_owner(
! CHECK: omp.taskgroup task_reduction(@_QQFlocal_ownermyred_i32
subroutine local_owner(x)
  integer :: x
  !$omp declare reduction(myred : integer : omp_out = omp_out + 10 * omp_in) &
  !$omp& initializer(omp_priv = 0)
  !$omp taskgroup task_reduction(myred: x)
  !$omp end taskgroup
end subroutine
