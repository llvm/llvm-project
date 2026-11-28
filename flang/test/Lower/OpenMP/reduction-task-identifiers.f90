! RUN: %flang_fc1 -fopenmp -fopenmp-version=50 -emit-hlfir -o - %s | FileCheck %s

! CHECK-DAG: omp.declare_reduction @_QQFnamed_ownermyred_i32
! CHECK-DAG: omp.declare_reduction @_QQFnamed_taskmyred_i32
! CHECK-DAG: omp.declare_reduction @_QQFnamed_targetmyred_i32
! CHECK-DAG: omp.declare_reduction @_QQFnamed_taskloopmyred_i32

! CHECK-LABEL: func.func @_QPnamed_owner(
! CHECK: omp.taskgroup task_reduction(@_QQFnamed_ownermyred_i32
subroutine named_owner(x)
  integer :: x
  !$omp declare reduction(myred : integer : omp_out = omp_out + 10 * omp_in) &
  !$omp& initializer(omp_priv = 0)
  !$omp taskgroup task_reduction(myred: x)
  !$omp end taskgroup
end subroutine

! CHECK-LABEL: func.func @_QPnamed_task(
! CHECK: omp.parallel reduction(task @_QQFnamed_taskmyred_i32
! CHECK: omp.task in_reduction(@_QQFnamed_taskmyred_i32
subroutine named_task(x)
  integer :: x
  !$omp declare reduction(myred : integer : omp_out = omp_out + 10 * omp_in) &
  !$omp& initializer(omp_priv = 0)
  !$omp parallel reduction(task, myred: x)
  !$omp task in_reduction(myred: x)
  x = x + 1
  !$omp end task
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPnamed_target(
! CHECK: omp.parallel reduction(task @_QQFnamed_targetmyred_i32
! CHECK: omp.target{{.*}}in_reduction(@_QQFnamed_targetmyred_i32
subroutine named_target(x)
  integer :: x
  !$omp declare reduction(myred : integer : omp_out = omp_out + 10 * omp_in) &
  !$omp& initializer(omp_priv = 0)
  !$omp parallel reduction(task, myred: x)
  !$omp target in_reduction(myred: x)
  x = x + 1
  !$omp end target
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPnamed_taskloop(
! CHECK: omp.parallel reduction(task @_QQFnamed_taskloopmyred_i32
! CHECK: omp.taskloop.context in_reduction(@_QQFnamed_taskloopmyred_i32
subroutine named_taskloop(x)
  integer :: x, i
  !$omp declare reduction(myred : integer : omp_out = omp_out + 10 * omp_in) &
  !$omp& initializer(omp_priv = 0)
  !$omp parallel reduction(task, myred: x)
  !$omp taskloop in_reduction(myred: x)
  do i = 1, 4
    x = x + 1
  end do
  !$omp end taskloop
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPpredefined_max(
! CHECK: omp.taskgroup task_reduction(@max_i32
subroutine predefined_max(x)
  integer :: x
  intrinsic :: max
  !$omp taskgroup task_reduction(max: x)
  !$omp end taskgroup
end subroutine

! CHECK-LABEL: func.func @_QPpredefined_max_implicit(
! CHECK: omp.taskgroup task_reduction(@max_i32
subroutine predefined_max_implicit(x)
  integer :: x
  !$omp taskgroup task_reduction(max: x)
  !$omp end taskgroup
end subroutine
