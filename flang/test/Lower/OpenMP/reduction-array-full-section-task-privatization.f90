! RUN: split-file %s %t
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/task.f90 | FileCheck %s --check-prefix=TASK --implicit-check-not=not\ yet\ implemented --implicit-check-not=Ea_firstprivate
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/taskloop.f90 | FileCheck %s --check-prefix=TASKLOOP --implicit-check-not=not\ yet\ implemented --implicit-check-not=Ea_firstprivate
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -mmlir --enable-delayed-privatization=false -o - %t/taskloop.f90 | FileCheck %s --check-prefix=TASKLOOP --implicit-check-not=not\ yet\ implemented --implicit-check-not=Ea_firstprivate

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

!--- task.f90
subroutine task_full_section(a)
  integer :: a(-2:1)
  !$omp taskgroup task_reduction(+: a(:))
  !$omp task in_reduction(+: a(:))
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
