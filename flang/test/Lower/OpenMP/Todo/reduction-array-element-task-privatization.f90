! RUN: split-file %s %t
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/task.f90 2>&1 | FileCheck %s --check-prefix=TASK
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/task.f90 2>&1 | FileCheck %s --check-prefix=TASK
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/taskloop-in.f90 2>&1 | FileCheck %s --check-prefix=TASKLOOP-IN
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/taskloop-in.f90 2>&1 | FileCheck %s --check-prefix=TASKLOOP-IN
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/taskloop-reduction.f90 2>&1 | FileCheck %s --check-prefix=TASKLOOP-REDUCTION
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/taskloop-reduction.f90 2>&1 | FileCheck %s --check-prefix=TASKLOOP-REDUCTION
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/task-section.f90 2>&1 | FileCheck %s --check-prefix=TASK-SECTION
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/task-section.f90 2>&1 | FileCheck %s --check-prefix=TASK-SECTION
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/task-rank-two-section.f90 2>&1 | FileCheck %s --check-prefix=TASK-RANK-TWO-SECTION
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/task-rank-two-section.f90 2>&1 | FileCheck %s --check-prefix=TASK-RANK-TWO-SECTION
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/taskloop-in-section.f90 2>&1 | FileCheck %s --check-prefix=TASKLOOP-IN-SECTION
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/taskloop-in-section.f90 2>&1 | FileCheck %s --check-prefix=TASKLOOP-IN-SECTION
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/taskloop-reduction-section.f90 2>&1 | FileCheck %s --check-prefix=TASKLOOP-REDUCTION-SECTION
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/taskloop-reduction-section.f90 2>&1 | FileCheck %s --check-prefix=TASKLOOP-REDUCTION-SECTION
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 --enable-delayed-privatization=false -o - %t/taskloop-in.f90 2>&1 | FileCheck %s --check-prefix=EAGER-TASKLOOP-IN
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -mmlir --enable-delayed-privatization=false -o - %t/taskloop-in.f90 2>&1 | FileCheck %s --check-prefix=EAGER-TASKLOOP-IN
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 --enable-delayed-privatization=false -o - %t/taskloop-reduction.f90 2>&1 | FileCheck %s --check-prefix=EAGER-TASKLOOP-REDUCTION
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -mmlir --enable-delayed-privatization=false -o - %t/taskloop-reduction.f90 2>&1 | FileCheck %s --check-prefix=EAGER-TASKLOOP-REDUCTION
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 --enable-delayed-privatization=false -o - %t/taskloop-in-section.f90 2>&1 | FileCheck %s --check-prefix=EAGER-TASKLOOP-IN-SECTION
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -mmlir --enable-delayed-privatization=false -o - %t/taskloop-in-section.f90 2>&1 | FileCheck %s --check-prefix=EAGER-TASKLOOP-IN-SECTION
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 --enable-delayed-privatization=false -o - %t/taskloop-reduction-section.f90 2>&1 | FileCheck %s --check-prefix=EAGER-TASKLOOP-REDUCTION-SECTION
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -mmlir --enable-delayed-privatization=false -o - %t/taskloop-reduction-section.f90 2>&1 | FileCheck %s --check-prefix=EAGER-TASKLOOP-REDUCTION-SECTION
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 --enable-delayed-privatization=false -o - %t/taskloop-udr-section.f90 2>&1 | FileCheck %s --check-prefix=EAGER-TASKLOOP-UDR-SECTION
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -mmlir --enable-delayed-privatization=false -o - %t/taskloop-udr-section.f90 2>&1 | FileCheck %s --check-prefix=EAGER-TASKLOOP-UDR-SECTION
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/task-cross-scope-bounds.f90 2>&1 | FileCheck %s --check-prefix=TASK-CROSS-SCOPE-BOUNDS
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/task-cross-scope-bounds.f90 2>&1 | FileCheck %s --check-prefix=TASK-CROSS-SCOPE-BOUNDS
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/taskloop-udr-shared-section.f90 2>&1 | FileCheck %s --check-prefix=TASKLOOP-UDR-SHARED-SECTION
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/taskloop-udr-shared-section.f90 2>&1 | FileCheck %s --check-prefix=TASKLOOP-UDR-SHARED-SECTION
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 --enable-delayed-privatization=false -o - %t/taskloop-udr-shared-section.f90 2>&1 | FileCheck %s --check-prefix=TASKLOOP-UDR-SHARED-SECTION
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -mmlir --enable-delayed-privatization=false -o - %t/taskloop-udr-shared-section.f90 2>&1 | FileCheck %s --check-prefix=TASKLOOP-UDR-SHARED-SECTION
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/taskloop-in-shared-section.f90 2>&1 | FileCheck %s --check-prefix=TASKLOOP-IN-SHARED-SECTION
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/taskloop-in-shared-section.f90 2>&1 | FileCheck %s --check-prefix=TASKLOOP-IN-SHARED-SECTION
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 --enable-delayed-privatization=false -o - %t/taskloop-in-shared-section.f90 2>&1 | FileCheck %s --check-prefix=TASKLOOP-IN-SHARED-SECTION
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -mmlir --enable-delayed-privatization=false -o - %t/taskloop-in-shared-section.f90 2>&1 | FileCheck %s --check-prefix=TASKLOOP-IN-SHARED-SECTION
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 --enable-delayed-privatization=false -o - %t/task-shared-element.f90 2>&1 | FileCheck %s --check-prefix=EAGER-TASK-SHARED-ELEMENT
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -mmlir --enable-delayed-privatization=false -o - %t/task-shared-element.f90 2>&1 | FileCheck %s --check-prefix=EAGER-TASK-SHARED-ELEMENT
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 --enable-delayed-privatization=false -o - %t/task-shared-full-section.f90 2>&1 | FileCheck %s --check-prefix=EAGER-TASK-SHARED-FULL-SECTION
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -mmlir --enable-delayed-privatization=false -o - %t/task-shared-full-section.f90 2>&1 | FileCheck %s --check-prefix=EAGER-TASK-SHARED-FULL-SECTION
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/task-udr-shared-element.f90 2>&1 | FileCheck %s --check-prefix=TASK-UDR-SHARED-ELEMENT
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/task-udr-shared-element.f90 2>&1 | FileCheck %s --check-prefix=TASK-UDR-SHARED-ELEMENT
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/taskgroup-udr-element.f90 2>&1 | FileCheck %s --check-prefix=TASKGROUP-UDR-ELEMENT
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/taskgroup-udr-element.f90 2>&1 | FileCheck %s --check-prefix=TASKGROUP-UDR-ELEMENT
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/taskloop-udr-shared-element.f90 2>&1 | FileCheck %s --check-prefix=TASKLOOP-UDR-SHARED-ELEMENT
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/taskloop-udr-shared-element.f90 2>&1 | FileCheck %s --check-prefix=TASKLOOP-UDR-SHARED-ELEMENT
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 --enable-delayed-privatization=false -o - %t/taskloop-udr-shared-element.f90 2>&1 | FileCheck %s --check-prefix=TASKLOOP-UDR-SHARED-ELEMENT
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -mmlir --enable-delayed-privatization=false -o - %t/taskloop-udr-shared-element.f90 2>&1 | FileCheck %s --check-prefix=TASKLOOP-UDR-SHARED-ELEMENT
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/taskloop-max-udr-shared-element.f90 2>&1 | FileCheck %s --check-prefix=TASKLOOP-MAX-UDR-SHARED-ELEMENT
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/taskloop-max-udr-shared-element.f90 2>&1 | FileCheck %s --check-prefix=TASKLOOP-MAX-UDR-SHARED-ELEMENT
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/taskloop-in-udr-shared-element.f90 2>&1 | FileCheck %s --check-prefix=TASKLOOP-IN-UDR-SHARED-ELEMENT
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/taskloop-in-udr-shared-element.f90 2>&1 | FileCheck %s --check-prefix=TASKLOOP-IN-UDR-SHARED-ELEMENT
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/task-shared-section.f90 2>&1 | FileCheck %s --check-prefix=TASK-SHARED-SECTION
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/task-shared-section.f90 2>&1 | FileCheck %s --check-prefix=TASK-SHARED-SECTION
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/target-element.f90 2>&1 | FileCheck %s --check-prefix=TARGET-ELEMENT
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %t/target-element.f90 2>&1 | FileCheck %s --check-prefix=TARGET-ELEMENT
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/parallel-task-udr-section.f90 2>&1 | FileCheck %s --check-prefix=PARALLEL-TASK-UDR-SECTION
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/parallel-task-udr-section.f90 2>&1 | FileCheck %s --check-prefix=PARALLEL-TASK-UDR-SECTION
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/sections-task-udr-section.f90 2>&1 | FileCheck %s --check-prefix=SECTIONS-TASK-UDR-SECTION
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/sections-task-udr-section.f90 2>&1 | FileCheck %s --check-prefix=SECTIONS-TASK-UDR-SECTION
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/scope-task-udr-section.f90 2>&1 | FileCheck %s --check-prefix=SCOPE-TASK-UDR-SECTION
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/scope-task-udr-section.f90 2>&1 | FileCheck %s --check-prefix=SCOPE-TASK-UDR-SECTION
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/do-task-udr-section.f90 2>&1 | FileCheck %s --check-prefix=DO-TASK-UDR-SECTION
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/do-task-udr-section.f90 2>&1 | FileCheck %s --check-prefix=DO-TASK-UDR-SECTION

! An array element or section in a task reduction and the implicitly
! firstprivate base array are represented by separate block arguments. Reject
! these constructs until lowering can bind references to the correct argument.

! TASK: not yet implemented: TASK construct with IN_REDUCTION of an array element or section whose base array is privatized
! TASKLOOP-IN: not yet implemented: TASKLOOP construct with IN_REDUCTION of an array element or section whose base array is privatized
! TASKLOOP-REDUCTION: not yet implemented: TASKLOOP construct with REDUCTION of an array element or section whose base array is privatized
! TASK-SECTION: not yet implemented: TASKGROUP construct with TASK_REDUCTION of a partial array section
! TASK-RANK-TWO-SECTION: not yet implemented: TASKGROUP construct with TASK_REDUCTION of a partial array section
! TASKLOOP-IN-SECTION: not yet implemented: TASKLOOP construct with IN_REDUCTION of an array element or section whose base array is privatized
! TASKLOOP-REDUCTION-SECTION: not yet implemented: TASKLOOP construct with REDUCTION of an array element or section whose base array is privatized
! EAGER-TASKLOOP-IN: not yet implemented: TASKLOOP construct with IN_REDUCTION of an array element or section whose base array is privatized
! EAGER-TASKLOOP-REDUCTION: not yet implemented: TASKLOOP construct with REDUCTION of an array element or section whose base array is privatized
! EAGER-TASKLOOP-IN-SECTION: not yet implemented: TASKLOOP construct with IN_REDUCTION of an array element or section whose base array is privatized
! EAGER-TASKLOOP-REDUCTION-SECTION: not yet implemented: TASKLOOP construct with REDUCTION of an array element or section whose base array is privatized
! EAGER-TASKLOOP-UDR-SECTION: not yet implemented: TASKLOOP construct with REDUCTION of an array element or section whose base array is privatized
! TASK-CROSS-SCOPE-BOUNDS: not yet implemented: TASKGROUP construct with TASK_REDUCTION of a partial array section
! TASKLOOP-UDR-SHARED-SECTION: not yet implemented: TASKLOOP construct with REDUCTION of a partial array section
! TASKLOOP-IN-SHARED-SECTION: not yet implemented: TASKLOOP construct with IN_REDUCTION of a partial array section
! EAGER-TASK-SHARED-ELEMENT: not yet implemented: TASK construct with IN_REDUCTION of an array element when delayed privatization is disabled
! EAGER-TASK-SHARED-FULL-SECTION: not yet implemented: TASK construct with IN_REDUCTION when delayed privatization is disabled
! TASK-UDR-SHARED-ELEMENT: not yet implemented: TASK construct with IN_REDUCTION of an array element using a user-defined reduction
! TASKGROUP-UDR-ELEMENT: not yet implemented: TASKGROUP construct with TASK_REDUCTION of an array element using a user-defined reduction
! TASKLOOP-UDR-SHARED-ELEMENT: not yet implemented: TASKLOOP construct with REDUCTION of an array element using a user-defined reduction
! TASKLOOP-MAX-UDR-SHARED-ELEMENT: not yet implemented: TASKLOOP construct with REDUCTION of an array element using a user-defined reduction
! TASKLOOP-IN-UDR-SHARED-ELEMENT: not yet implemented: TASKLOOP construct with IN_REDUCTION of an array element using a user-defined reduction
! TASK-SHARED-SECTION: not yet implemented: TASK construct with IN_REDUCTION of a partial array section
! TARGET-ELEMENT: not yet implemented: TARGET construct with IN_REDUCTION of an array element
! PARALLEL-TASK-UDR-SECTION: not yet implemented: REDUCTION with TASK modifier of a partial array section
! SECTIONS-TASK-UDR-SECTION: not yet implemented: REDUCTION with TASK modifier of a partial array section
! SCOPE-TASK-UDR-SECTION: not yet implemented: REDUCTION with TASK modifier of a partial array section
! DO-TASK-UDR-SECTION: not yet implemented: REDUCTION with TASK modifier of a partial array section

!--- task.f90
subroutine task_reduction_element(a)
  integer :: a(4)
  !$omp taskgroup task_reduction(+: a(2))
  !$omp task in_reduction(+: a(2))
  a(2) = a(2) + 1
  !$omp end task
  !$omp end taskgroup
end subroutine

!--- task-shared-element.f90
subroutine task_in_reduction_shared_element(a)
  integer :: a(4)
  !$omp parallel shared(a)
  !$omp single
  !$omp task in_reduction(+: a(2))
  a(2) = a(2) + 1
  !$omp end task
  !$omp end single
  !$omp end parallel
end subroutine

!--- task-shared-full-section.f90
subroutine task_in_reduction_shared_full_section(a)
  integer :: a(4)
  !$omp taskgroup task_reduction(+: a(:))
  !$omp task shared(a) in_reduction(+: a(:))
  a(:) = a(:) + 1
  !$omp end task
  !$omp end taskgroup
end subroutine

!--- task-udr-shared-element.f90
subroutine task_in_reduction_udr_shared_element(a)
  integer :: a(4)
  !$omp declare reduction(+: integer : omp_out = omp_out + omp_in) &
  !$omp& initializer(omp_priv = 1)
  !$omp task shared(a) in_reduction(+: a(2))
  a(2) = a(2) + 1
  !$omp end task
end subroutine

!--- taskgroup-udr-element.f90
subroutine taskgroup_udr_element(a)
  integer :: a(4)
  !$omp declare reduction(+: integer : omp_out = omp_out + omp_in) &
  !$omp& initializer(omp_priv = 1)
  !$omp taskgroup task_reduction(+: a(2))
  a(2) = a(2) + 1
  !$omp end taskgroup
end subroutine

!--- task-shared-section.f90
subroutine task_in_reduction_shared_section(a)
  integer :: a(4)
  !$omp task shared(a) in_reduction(+: a(2:3))
  a(2:3) = a(2:3) + 1
  !$omp end task
end subroutine

!--- target-element.f90
subroutine target_in_reduction_element(a)
  integer :: a(4)
  !$omp target in_reduction(+: a(2)) map(tofrom: a)
  a(2) = a(2) + 1
  !$omp end target
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

!--- task-rank-two-section.f90
subroutine task_reduction_rank_two_section(a)
  integer :: a(4, 4)
  !$omp taskgroup task_reduction(+: a(:, 2))
  !$omp task in_reduction(+: a(:, 2))
  a(:, 2) = a(:, 2) + 1
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

!--- taskloop-udr-section.f90
subroutine taskloop_udr_section(a)
  integer :: a(4), i
  !$omp declare reduction(myred : integer : omp_out = omp_out + omp_in) &
  !$omp& initializer(omp_priv = 1)
  !$omp taskloop reduction(myred : a(2:3))
  do i = 1, 1
    a(2:3) = a(2:3) + i
  end do
end subroutine

!--- taskloop-udr-shared-section.f90
subroutine taskloop_udr_shared_section(a)
  integer :: a(4), i
  !$omp declare reduction(myred : integer : omp_out = omp_out + omp_in) &
  !$omp& initializer(omp_priv = 1)
  !$omp parallel shared(a)
  !$omp single
  !$omp taskloop reduction(myred : a(2:3))
  do i = 1, 1
    a(2:3) = a(2:3) + i
  end do
  !$omp end single
  !$omp end parallel
end subroutine

!--- taskloop-udr-shared-element.f90
subroutine taskloop_udr_shared_element(a)
  integer :: a(4), i
  !$omp declare reduction(myred : integer : omp_out = omp_out + omp_in) &
  !$omp& initializer(omp_priv = 1)
  !$omp parallel shared(a)
  !$omp single
  !$omp taskloop reduction(myred : a(2))
  do i = 1, 1
    a(2) = a(2) + i
  end do
  !$omp end single
  !$omp end parallel
end subroutine

!--- taskloop-max-udr-shared-element.f90
subroutine taskloop_max_udr_shared_element(a)
  integer :: a(4), i
  intrinsic :: max
  !$omp declare reduction(max : integer : omp_out = omp_out + omp_in) &
  !$omp& initializer(omp_priv = 1)
  !$omp parallel shared(a)
  !$omp single
  !$omp taskloop reduction(max : a(2))
  do i = 1, 1
    a(2) = a(2) + i
  end do
  !$omp end single
  !$omp end parallel
end subroutine

!--- taskloop-in-udr-shared-element.f90
subroutine taskloop_in_reduction_udr_shared_element(a)
  integer :: a(4), i
  !$omp declare reduction(+: integer : omp_out = omp_out + omp_in) &
  !$omp& initializer(omp_priv = 1)
  !$omp parallel shared(a)
  !$omp single
  !$omp taskloop in_reduction(+: a(2))
  do i = 1, 1
    a(2) = a(2) + i
  end do
  !$omp end single
  !$omp end parallel
end subroutine

!--- taskloop-in-shared-section.f90
subroutine taskloop_in_reduction_shared_section(a)
  integer :: a(4, 4), i
  !$omp parallel shared(a)
  !$omp single
  !$omp taskloop in_reduction(+: a(:, 2))
  do i = 1, 1
    a(:, 2) = a(:, 2) + i
  end do
  !$omp end single
  !$omp end parallel
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

!--- task-cross-scope-bounds.f90
subroutine task_cross_scope_bounds(n)
  integer :: n
  integer :: a(n)

contains
  subroutine inner(m)
    integer :: m
    !$omp declare reduction(+ : integer : omp_out = omp_out + omp_in) &
    !$omp& initializer(omp_priv = 1)
    !$omp taskgroup task_reduction(+: a(1:m))
    !$omp task in_reduction(+: a(1:m))
    a(1:m) = a(1:m) + 1
    !$omp end task
    !$omp end taskgroup
  end subroutine
end subroutine

!--- parallel-task-udr-section.f90
subroutine parallel_task_udr_section(a)
  integer :: a(4)
  !$omp declare reduction(+ : integer : omp_out = omp_out + omp_in) &
  !$omp& initializer(omp_priv = 1)
  !$omp parallel reduction(task, + : a(2:3))
  !$omp target map(tofrom: a) in_reduction(+ : a(2:3))
  a(2:3) = a(2:3) + 1
  !$omp end target
  !$omp end parallel
end subroutine

!--- sections-task-udr-section.f90
subroutine sections_task_udr_section(a)
  integer :: a(4)
  !$omp declare reduction(+ : integer : omp_out = omp_out + omp_in) &
  !$omp& initializer(omp_priv = 1)
  !$omp sections reduction(task, + : a(2:3))
  !$omp section
  a(2:3) = a(2:3) + 1
  !$omp end sections
end subroutine

!--- scope-task-udr-section.f90
subroutine scope_task_udr_section(a)
  integer :: a(4)
  !$omp declare reduction(+ : integer : omp_out = omp_out + omp_in) &
  !$omp& initializer(omp_priv = 1)
  !$omp scope reduction(task, + : a(2:3))
  a(2:3) = a(2:3) + 1
  !$omp end scope
end subroutine

!--- do-task-udr-section.f90
subroutine do_task_udr_section(a)
  integer :: a(4), i
  !$omp declare reduction(+ : integer : omp_out = omp_out + omp_in) &
  !$omp& initializer(omp_priv = 1)
  !$omp do reduction(task, + : a(2:3))
  do i = 1, 1
    a(2:3) = a(2:3) + i
  end do
  !$omp end do
end subroutine
