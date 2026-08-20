! RUN: split-file %s %t
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/task.f90 | FileCheck %s --check-prefix=TASK --implicit-check-not="not yet implemented" --implicit-check-not=Ea_firstprivate
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/task-default.f90 | FileCheck %s --check-prefix=TASK-DEFAULT --implicit-check-not="not yet implemented" --implicit-check-not=Ea_firstprivate --implicit-check-not=Ea_private
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/taskloop.f90 | FileCheck %s --check-prefix=TASKLOOP --implicit-check-not="not yet implemented" --implicit-check-not=Ea_firstprivate
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -mmlir --enable-delayed-privatization=false -o - %t/taskloop.f90 | FileCheck %s --check-prefix=TASKLOOP --implicit-check-not="not yet implemented" --implicit-check-not=Ea_firstprivate

! A full-extent section uses the same descriptor as its base array. Check that
! it is bound only to the reduction argument, rather than also being captured
! as an implicit firstprivate object.

! TASK-LABEL: func.func @_QPtask_full_section
! TASK: omp.taskgroup task_reduction(byref @add_reduction_byref_box_4xi32 {{.*}} -> %[[TASKGROUP_ARG:arg[0-9]+]] : !fir.ref<!fir.box<!fir.array<4xi32>>>)
! TASK: %[[TASKGROUP_DECL:.*]]:2 = hlfir.declare %[[TASKGROUP_ARG]]
! TASK: omp.task in_reduction(byref @add_reduction_byref_box_4xi32 %[[TASKGROUP_DECL]]#0 -> %[[TASK_ARG:arg[0-9]+]] : !fir.ref<!fir.box<!fir.array<4xi32>>>)
! TASK: %[[TASK_DECL:.*]]:2 = hlfir.declare %[[TASK_ARG]]
! TASK: %[[TASK_BOX:.*]] = fir.load %[[TASK_DECL]]#0
! TASK: %[[TASK_SECTION:.*]] = hlfir.designate %[[TASK_BOX]]
! TASK: hlfir.elemental
! TASK: %[[TASK_ELEMENT:.*]] = hlfir.designate %[[TASK_SECTION]]
! TASK: %[[TASK_VALUE:.*]] = fir.load %[[TASK_ELEMENT]]
! TASK: arith.addi %[[TASK_VALUE]]

! TASK-LABEL: func.func @_QPtask_rank_two_full_section
! TASK: omp.taskgroup task_reduction(byref @add_reduction_byref_box_4x4xi32 {{.*}} -> %[[RANK_TWO_TASKGROUP_ARG:arg[0-9]+]] : !fir.ref<!fir.box<!fir.array<4x4xi32>>>)
! TASK: %[[RANK_TWO_TASKGROUP_DECL:.*]]:2 = hlfir.declare %[[RANK_TWO_TASKGROUP_ARG]]
! TASK: omp.task in_reduction(byref @add_reduction_byref_box_4x4xi32 %[[RANK_TWO_TASKGROUP_DECL]]#0 -> %[[RANK_TWO_TASK_ARG:arg[0-9]+]] : !fir.ref<!fir.box<!fir.array<4x4xi32>>>)
! TASK: %[[RANK_TWO_TASK_DECL:.*]]:2 = hlfir.declare %[[RANK_TWO_TASK_ARG]]
! TASK: %[[RANK_TWO_TASK_BOX:.*]] = fir.load %[[RANK_TWO_TASK_DECL]]#0
! TASK: %[[RANK_TWO_TASK_SECTION:.*]] = hlfir.designate %[[RANK_TWO_TASK_BOX]]
! TASK: hlfir.elemental
! TASK: %[[RANK_TWO_TASK_ELEMENT:.*]] = hlfir.designate %[[RANK_TWO_TASK_SECTION]]
! TASK: %[[RANK_TWO_TASK_VALUE:.*]] = fir.load %[[RANK_TWO_TASK_ELEMENT]]
! TASK: arith.addi %[[RANK_TWO_TASK_VALUE]]

! TASK-LABEL: func.func @_QPtask_explicit_full_section
! TASK: omp.taskgroup task_reduction(byref @add_reduction_byref_box_4xi32
! TASK: omp.task in_reduction(byref @add_reduction_byref_box_4xi32

! TASK-LABEL: func.func @_QPtask_dynamic_full_section
! TASK: omp.taskgroup task_reduction(byref @add_reduction_byref_box_Uxi32 {{.*}} -> %[[DYNAMIC_TASKGROUP_ARG:arg[0-9]+]] : !fir.ref<!fir.box<!fir.array<?xi32>>>)
! TASK: %[[DYNAMIC_TASKGROUP_DECL:.*]]:2 = hlfir.declare %[[DYNAMIC_TASKGROUP_ARG]]
! TASK: omp.task in_reduction(byref @add_reduction_byref_box_Uxi32 %[[DYNAMIC_TASKGROUP_DECL]]#0 -> %[[DYNAMIC_TASK_ARG:arg[0-9]+]] : !fir.ref<!fir.box<!fir.array<?xi32>>>)
! TASK: %[[DYNAMIC_TASK_DECL:.*]]:2 = hlfir.declare %[[DYNAMIC_TASK_ARG]]
! TASK: %[[DYNAMIC_TASK_BOX:.*]] = fir.load %[[DYNAMIC_TASK_DECL]]#0
! TASK: hlfir.designate %[[DYNAMIC_TASK_BOX]]

! TASK-DEFAULT-LABEL: func.func @_QPtask_default_firstprivate_full_section
! TASK-DEFAULT: omp.task in_reduction(byref @add_reduction_byref_box_4xi32 {{.*}} -> %[[FIRSTPRIVATE_ARG:arg[0-9]+]] : !fir.ref<!fir.box<!fir.array<4xi32>>>)
! TASK-DEFAULT: %[[FIRSTPRIVATE_DECL:.*]]:2 = hlfir.declare %[[FIRSTPRIVATE_ARG]]
! TASK-DEFAULT: %[[FIRSTPRIVATE_BOX:.*]] = fir.load %[[FIRSTPRIVATE_DECL]]#0
! TASK-DEFAULT: %[[FIRSTPRIVATE_SECTION:.*]] = hlfir.designate %[[FIRSTPRIVATE_BOX]]
! TASK-DEFAULT: hlfir.elemental
! TASK-DEFAULT: %[[FIRSTPRIVATE_ELEMENT:.*]] = hlfir.designate %[[FIRSTPRIVATE_SECTION]]
! TASK-DEFAULT: %[[FIRSTPRIVATE_VALUE:.*]] = fir.load %[[FIRSTPRIVATE_ELEMENT]]
! TASK-DEFAULT: arith.addi %[[FIRSTPRIVATE_VALUE]]

! TASK-DEFAULT-LABEL: func.func @_QPtask_default_private_full_section
! TASK-DEFAULT: omp.task in_reduction(byref @add_reduction_byref_box_4xi32 {{.*}} -> %[[PRIVATE_ARG:arg[0-9]+]] : !fir.ref<!fir.box<!fir.array<4xi32>>>)
! TASK-DEFAULT: %[[PRIVATE_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_ARG]]
! TASK-DEFAULT: %[[PRIVATE_BOX:.*]] = fir.load %[[PRIVATE_DECL]]#0
! TASK-DEFAULT: %[[PRIVATE_SECTION:.*]] = hlfir.designate %[[PRIVATE_BOX]]
! TASK-DEFAULT: hlfir.elemental
! TASK-DEFAULT: %[[PRIVATE_ELEMENT:.*]] = hlfir.designate %[[PRIVATE_SECTION]]
! TASK-DEFAULT: %[[PRIVATE_VALUE:.*]] = fir.load %[[PRIVATE_ELEMENT]]
! TASK-DEFAULT: arith.addi %[[PRIVATE_VALUE]]

! TASKLOOP-LABEL: func.func @_QPtaskloop_in_full_section
! TASKLOOP: omp.taskloop.context in_reduction(byref @add_reduction_byref_box_4xi32 {{.*}} -> %[[IN_ARG:arg[0-9]+]] : !fir.ref<!fir.box<!fir.array<4xi32>>>)
! TASKLOOP: %[[IN_DECL:.*]]:2 = hlfir.declare %[[IN_ARG]]
! TASKLOOP: %[[IN_BOX:.*]] = fir.load %[[IN_DECL]]#0
! TASKLOOP: %[[IN_SECTION:.*]] = hlfir.designate %[[IN_BOX]]
! TASKLOOP: hlfir.elemental
! TASKLOOP: %[[IN_ELEMENT:.*]] = hlfir.designate %[[IN_SECTION]]
! TASKLOOP: %[[IN_VALUE:.*]] = fir.load %[[IN_ELEMENT]]
! TASKLOOP: arith.addi %[[IN_VALUE]]

! TASKLOOP-LABEL: func.func @_QPtaskloop_reduction_full_section
! TASKLOOP: omp.taskloop.context {{.*}}reduction(byref @add_reduction_byref_box_4xi32 {{.*}} -> %[[RED_ARG:arg[0-9]+]] : !fir.ref<!fir.box<!fir.array<4xi32>>>)
! TASKLOOP: %[[RED_DECL:.*]]:2 = hlfir.declare %[[RED_ARG]]
! TASKLOOP: %[[RED_BOX:.*]] = fir.load %[[RED_DECL]]#0
! TASKLOOP: %[[RED_SECTION:.*]] = hlfir.designate %[[RED_BOX]]
! TASKLOOP: hlfir.elemental
! TASKLOOP: %[[RED_ELEMENT:.*]] = hlfir.designate %[[RED_SECTION]]
! TASKLOOP: %[[RED_VALUE:.*]] = fir.load %[[RED_ELEMENT]]
! TASKLOOP: arith.addi %[[RED_VALUE]]

! TASKLOOP-LABEL: func.func @_QPtaskloop_udr_full_section
! TASKLOOP: omp.taskloop.context {{.*}}reduction(byref @_QQFtaskloop_udr_full_sectionmyred_byref_box_4xi32 {{.*}} -> %[[UDR_ARG:arg[0-9]+]] : !fir.ref<!fir.box<!fir.array<4xi32>>>)
! TASKLOOP: %[[UDR_DECL:.*]]:2 = hlfir.declare %[[UDR_ARG]]
! TASKLOOP: %[[UDR_BOX:.*]] = fir.load %[[UDR_DECL]]#0
! TASKLOOP: %[[UDR_SECTION:.*]] = hlfir.designate %[[UDR_BOX]]
! TASKLOOP: hlfir.elemental
! TASKLOOP: %[[UDR_ELEMENT:.*]] = hlfir.designate %[[UDR_SECTION]]
! TASKLOOP: %[[UDR_VALUE:.*]] = fir.load %[[UDR_ELEMENT]]
! TASKLOOP: arith.addi %[[UDR_VALUE]]

! TASKLOOP-LABEL: func.func @_QPtaskloop_explicit_full_section
! TASKLOOP: omp.taskloop.context {{.*}}reduction(byref @add_reduction_byref_box_4xi32

! TASKLOOP-LABEL: func.func @_QPtaskloop_rank_two_explicit_full_section
! TASKLOOP: omp.taskloop.context {{.*}}reduction(byref @add_reduction_byref_box_4x4xi32

!--- task.f90
subroutine task_full_section(a)
  integer :: a(-2:1)
  !$omp taskgroup task_reduction(+: a(:))
  !$omp task in_reduction(+: a(:))
  a(:) = a(:) + 1
  !$omp end task
  !$omp end taskgroup
end subroutine

subroutine task_rank_two_full_section(a)
  integer :: a(-2:1, -1:2)
  !$omp taskgroup task_reduction(+: a(:, :))
  !$omp task in_reduction(+: a(:, :))
  a(:, :) = a(:, :) + 1
  !$omp end task
  !$omp end taskgroup
end subroutine

subroutine task_explicit_full_section(a)
  integer :: a(-2:1)
  !$omp taskgroup task_reduction(+: a(-2:1))
  !$omp task in_reduction(+: a(-2:1))
  a(-2:1) = a(-2:1) + 1
  !$omp end task
  !$omp end taskgroup
end subroutine

subroutine task_dynamic_full_section(a)
  integer :: a(:)
  !$omp taskgroup task_reduction(+: a(lbound(a, 1):ubound(a, 1)))
  !$omp task shared(a) in_reduction(+: a(lbound(a, 1):ubound(a, 1)))
  a(lbound(a, 1):ubound(a, 1)) = a(lbound(a, 1):ubound(a, 1)) + 1
  !$omp end task
  !$omp end taskgroup
end subroutine

!--- task-default.f90
subroutine task_default_firstprivate_full_section(a)
  integer :: a(-2:1)
  !$omp taskgroup task_reduction(+: a(:))
  !$omp task default(firstprivate) in_reduction(+: a(:))
  a(:) = a(:) + 1
  !$omp end task
  !$omp end taskgroup
end subroutine

subroutine task_default_private_full_section(a)
  integer :: a(-2:1)
  !$omp taskgroup task_reduction(+: a(:))
  !$omp task default(private) in_reduction(+: a(:))
  a(:) = a(:) + 1
  !$omp end task
  !$omp end taskgroup
end subroutine

!--- taskloop.f90
subroutine taskloop_in_full_section(a, n)
  integer :: a(-2:1), n
  !$omp taskloop in_reduction(+: a(:))
  do i = 1, n
    a(:) = a(:) + i
  end do
end subroutine

subroutine taskloop_reduction_full_section(a, n)
  integer :: a(-2:1), n
  !$omp taskloop reduction(+: a(:))
  do i = 1, n
    a(:) = a(:) + i
  end do
end subroutine

subroutine taskloop_udr_full_section(a)
  integer :: a(-2:1), i
  !$omp declare reduction(myred : integer : omp_out = omp_out + omp_in) &
  !$omp& initializer(omp_priv = 1)
  !$omp taskloop reduction(myred : a(:))
  do i = 1, 1
    a(:) = a(:) + i
  end do
end subroutine

subroutine taskloop_explicit_full_section(a)
  integer :: a(-2:1), i
  !$omp parallel shared(a)
  !$omp single
  !$omp taskloop reduction(+: a(-2:1))
  do i = 1, 1
    a(-2:1) = a(-2:1) + i
  end do
  !$omp end single
  !$omp end parallel
end subroutine

subroutine taskloop_rank_two_explicit_full_section(a)
  integer :: a(-2:1, -1:2), i
  !$omp parallel shared(a)
  !$omp single
  !$omp taskloop reduction(+: a(-2:1, -1:2))
  do i = 1, 1
    a(-2:1, -1:2) = a(-2:1, -1:2) + i
  end do
  !$omp end single
  !$omp end parallel
end subroutine
