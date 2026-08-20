! RUN: split-file %s %t
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/task.f90 | FileCheck %s --check-prefix=TASK --implicit-check-not="not yet implemented"
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/taskgroup.f90 | FileCheck %s --check-prefix=TASKGROUP --implicit-check-not="not yet implemented"
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/taskloop.f90 | FileCheck %s --check-prefix=TASKLOOP --implicit-check-not="not yet implemented"
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -mmlir --enable-delayed-privatization=false -o - %t/taskloop.f90 | FileCheck %s --check-prefix=TASKLOOP --implicit-check-not="not yet implemented"

! A single-element reduction is lowered as a reduction of its whole base array
! when that path is otherwise supported. Check that the generated reduction
! arguments therefore use the base array descriptor.

! TASK-LABEL: func.func @_QPtask_element
! TASK: omp.taskgroup task_reduction(byref @add_reduction_byref_box_4xi32 {{.*}} -> %[[TASKGROUP_ARG:arg[0-9]+]] : !fir.ref<!fir.box<!fir.array<4xi32>>>)
! TASK: %[[TASKGROUP_DECL:.*]]:2 = hlfir.declare %[[TASKGROUP_ARG]]
! TASK: omp.task in_reduction(byref @add_reduction_byref_box_4xi32 %[[TASKGROUP_DECL]]#0 -> %[[TASK_ARG:arg[0-9]+]] : !fir.ref<!fir.box<!fir.array<4xi32>>>)
! TASK: %[[TASK_DECL:.*]]:2 = hlfir.declare %[[TASK_ARG]]
! TASK: %[[TASK_BOX:.*]] = fir.load %[[TASK_DECL]]#0
! TASK: hlfir.designate %[[TASK_BOX]]

! TASKGROUP-LABEL: func.func @_QPtaskgroup_element
! TASKGROUP: omp.taskgroup task_reduction(byref @add_reduction_byref_box_4xi32 {{.*}} -> %[[TASKGROUP_ONLY_ARG:arg[0-9]+]] : !fir.ref<!fir.box<!fir.array<4xi32>>>)
! TASKGROUP: %[[TASKGROUP_ONLY_DECL:.*]]:2 = hlfir.declare %[[TASKGROUP_ONLY_ARG]]
! TASKGROUP: %[[TASKGROUP_ONLY_BOX:.*]] = fir.load %[[TASKGROUP_ONLY_DECL]]#0
! TASKGROUP: hlfir.designate %[[TASKGROUP_ONLY_BOX]]

! TASKLOOP-LABEL: func.func @_QPtaskloop_in_element
! TASKLOOP: omp.taskgroup task_reduction(byref @add_reduction_byref_box_4xi32 {{.*}} -> %[[TASKLOOP_GROUP_ARG:arg[0-9]+]] : !fir.ref<!fir.box<!fir.array<4xi32>>>)
! TASKLOOP: %[[TASKLOOP_GROUP_DECL:.*]]:2 = hlfir.declare %[[TASKLOOP_GROUP_ARG]]
! TASKLOOP: omp.taskloop.context in_reduction(byref @add_reduction_byref_box_4xi32 %[[TASKLOOP_GROUP_DECL]]#0 -> %[[TASKLOOP_ARG:arg[0-9]+]] : !fir.ref<!fir.box<!fir.array<4xi32>>>)
! TASKLOOP: %[[TASKLOOP_DECL:.*]]:2 = hlfir.declare %[[TASKLOOP_ARG]]
! TASKLOOP: %[[TASKLOOP_BOX:.*]] = fir.load %[[TASKLOOP_DECL]]#0
! TASKLOOP: hlfir.designate %[[TASKLOOP_BOX]]

! TASKLOOP-LABEL: func.func @_QPtaskloop_reduction_element
! TASKLOOP: omp.taskloop.context {{.*}}reduction(byref @add_reduction_byref_box_4xi32

!--- task.f90
subroutine task_element(a)
  integer :: a(4)
  !$omp parallel shared(a)
  !$omp single
  !$omp taskgroup task_reduction(+: a(2))
  !$omp task in_reduction(+: a(2))
  a(2) = a(2) + 1
  !$omp end task
  !$omp end taskgroup
  !$omp end single
  !$omp end parallel
end subroutine

!--- taskgroup.f90
subroutine taskgroup_element(a)
  integer :: a(4)
  !$omp taskgroup task_reduction(+: a(2))
  a(2) = a(2) + 1
  !$omp end taskgroup
end subroutine

!--- taskloop.f90
subroutine taskloop_in_element(a)
  integer :: a(4), i
  !$omp parallel shared(a)
  !$omp single
  !$omp taskgroup task_reduction(+: a(2))
  !$omp taskloop in_reduction(+: a(2))
  do i = 1, 1
    a(2) = a(2) + i
  end do
  !$omp end taskgroup
  !$omp end single
  !$omp end parallel
end subroutine

subroutine taskloop_reduction_element(a)
  integer :: a(4), i
  !$omp parallel shared(a)
  !$omp single
  !$omp taskloop reduction(+: a(2))
  do i = 1, 1
    a(2) = a(2) + i
  end do
  !$omp end single
  !$omp end parallel
end subroutine
