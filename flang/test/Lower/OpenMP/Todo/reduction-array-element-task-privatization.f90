! RUN: split-file %s %t
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/task.f90 2>&1 | FileCheck %s --check-prefix=TASK
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/task.f90 2>&1 | FileCheck %s --check-prefix=TASK
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/taskloop-in.f90 2>&1 | FileCheck %s --check-prefix=TASKLOOP-IN
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/taskloop-in.f90 2>&1 | FileCheck %s --check-prefix=TASKLOOP-IN
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/taskloop-reduction.f90 2>&1 | FileCheck %s --check-prefix=TASKLOOP-REDUCTION
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/taskloop-reduction.f90 2>&1 | FileCheck %s --check-prefix=TASKLOOP-REDUCTION
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/task-section.f90 2>&1 | FileCheck %s --check-prefix=TASK-SECTION
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/task-section.f90 2>&1 | FileCheck %s --check-prefix=TASK-SECTION
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/taskloop-in-section.f90 2>&1 | FileCheck %s --check-prefix=TASKLOOP-IN-SECTION
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/taskloop-in-section.f90 2>&1 | FileCheck %s --check-prefix=TASKLOOP-IN-SECTION
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/taskloop-reduction-section.f90 2>&1 | FileCheck %s --check-prefix=TASKLOOP-REDUCTION-SECTION
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/taskloop-reduction-section.f90 2>&1 | FileCheck %s --check-prefix=TASKLOOP-REDUCTION-SECTION
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 --enable-delayed-privatization=false -o - %t/taskloop-in.f90 2>&1 | FileCheck %s --check-prefix=EAGER-TASKLOOP-IN
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -mmlir --enable-delayed-privatization=false -o - %t/taskloop-in.f90 2>&1 | FileCheck %s --check-prefix=EAGER-TASKLOOP-IN
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 --enable-delayed-privatization=false -o - %t/taskloop-reduction.f90 2>&1 | FileCheck %s --check-prefix=EAGER-TASKLOOP-REDUCTION
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -mmlir --enable-delayed-privatization=false -o - %t/taskloop-reduction.f90 2>&1 | FileCheck %s --check-prefix=EAGER-TASKLOOP-REDUCTION
! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=50 --enable-delayed-privatization=false -o %t/eager-in-section-bbc.mlir %t/taskloop-in-section.f90
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -mmlir --enable-delayed-privatization=false -o %t/eager-in-section-fc1.mlir %t/taskloop-in-section.f90
! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=50 --enable-delayed-privatization=false -o %t/eager-reduction-section-bbc.mlir %t/taskloop-reduction-section.f90
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -mmlir --enable-delayed-privatization=false -o %t/eager-reduction-section-fc1.mlir %t/taskloop-reduction-section.f90

! An array element or section in a task reduction and the implicitly
! firstprivate base array are represented by separate block arguments. Reject
! these constructs until lowering can bind references to the correct argument.

! TASK: not yet implemented: TASK construct with IN_REDUCTION of an array element or section whose base array is privatized
! TASKLOOP-IN: not yet implemented: TASKLOOP construct with IN_REDUCTION of an array element or section whose base array is privatized
! TASKLOOP-REDUCTION: not yet implemented: TASKLOOP construct with REDUCTION of an array element or section whose base array is privatized
! TASK-SECTION: not yet implemented: TASK construct with IN_REDUCTION of an array element or section whose base array is privatized
! TASKLOOP-IN-SECTION: not yet implemented: TASKLOOP construct with IN_REDUCTION of an array element or section whose base array is privatized
! TASKLOOP-REDUCTION-SECTION: not yet implemented: TASKLOOP construct with REDUCTION of an array element or section whose base array is privatized
! EAGER-TASKLOOP-IN: not yet implemented: TASKLOOP construct with IN_REDUCTION of an array element whose base array is privatized
! EAGER-TASKLOOP-REDUCTION: not yet implemented: TASKLOOP construct with REDUCTION of an array element whose base array is privatized

!--- task.f90
subroutine task_reduction_element(a)
  integer :: a(4)
  !$omp taskgroup task_reduction(+: a(2))
  !$omp task in_reduction(+: a(2))
  a(2) = a(2) + 1
  !$omp end task
  !$omp end taskgroup
end subroutine

!--- task-section.f90
subroutine task_reduction_section(a)
  integer :: a(4)
  !$omp taskgroup task_reduction(+: a(2:3))
  !$omp task in_reduction(+: a(2:3))
  a(2:3) = a(2:3) + 1
  !$omp end task
  !$omp end taskgroup
end subroutine

!--- taskloop-in-section.f90
subroutine taskloop_in_reduction_section(a, n)
  integer :: a(4), n
  !$omp taskloop in_reduction(+: a(2:3))
  do i = 1, n
    a(2:3) = a(2:3) + i
  end do
end subroutine

!--- taskloop-reduction-section.f90
subroutine taskloop_reduction_section(a, n)
  integer :: a(4), n
  !$omp taskloop reduction(+: a(2:3))
  do i = 1, n
    a(2:3) = a(2:3) + i
  end do
end subroutine

!--- taskloop-in.f90
subroutine taskloop_in_reduction_element(a, n)
  integer :: a(4), n
  !$omp taskloop in_reduction(+: a(2))
  do i = 1, n
    a(2) = a(2) + i
  end do
end subroutine

!--- taskloop-reduction.f90
subroutine taskloop_reduction_element(a, n)
  integer :: a(4), n
  !$omp taskloop reduction(+: a(2))
  do i = 1, n
    a(2) = a(2) + i
  end do
end subroutine
