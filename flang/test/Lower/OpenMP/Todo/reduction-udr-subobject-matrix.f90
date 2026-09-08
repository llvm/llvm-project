! RUN: split-file %s %t
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/parallel-section.f90 2>&1 | FileCheck %s --check-prefix=SECTION
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/parallel-section.f90 2>&1 | FileCheck %s --check-prefix=SECTION
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/do-section.f90 2>&1 | FileCheck %s --check-prefix=SECTION
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/do-section.f90 2>&1 | FileCheck %s --check-prefix=SECTION
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/sections-section.f90 2>&1 | FileCheck %s --check-prefix=SECTION
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/sections-section.f90 2>&1 | FileCheck %s --check-prefix=SECTION
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/scope-section.f90 2>&1 | FileCheck %s --check-prefix=SECTION
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/scope-section.f90 2>&1 | FileCheck %s --check-prefix=SECTION
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/simd-element.f90 2>&1 | FileCheck %s --check-prefix=ELEMENT
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/simd-element.f90 2>&1 | FileCheck %s --check-prefix=ELEMENT
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/simd-section.f90 2>&1 | FileCheck %s --check-prefix=SECTION
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/simd-section.f90 2>&1 | FileCheck %s --check-prefix=SECTION
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/teams-element.f90 2>&1 | FileCheck %s --check-prefix=ELEMENT
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/teams-element.f90 2>&1 | FileCheck %s --check-prefix=ELEMENT
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/teams-section.f90 2>&1 | FileCheck %s --check-prefix=SECTION
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/teams-section.f90 2>&1 | FileCheck %s --check-prefix=SECTION
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/loop-element.f90 2>&1 | FileCheck %s --check-prefix=ELEMENT
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/loop-element.f90 2>&1 | FileCheck %s --check-prefix=ELEMENT
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/loop-section.f90 2>&1 | FileCheck %s --check-prefix=SECTION
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/loop-section.f90 2>&1 | FileCheck %s --check-prefix=SECTION
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/in-reduction-section.f90 2>&1 | FileCheck %s --check-prefix=IN-SECTION
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/in-reduction-section.f90 2>&1 | FileCheck %s --check-prefix=IN-SECTION
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/task-reduction-section.f90 2>&1 | FileCheck %s --check-prefix=TASK-SECTION
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/task-reduction-section.f90 2>&1 | FileCheck %s --check-prefix=TASK-SECTION
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/complex-part.f90 2>&1 | FileCheck %s --check-prefix=ELEMENT
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/complex-part.f90 2>&1 | FileCheck %s --check-prefix=ELEMENT
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/complex-array-part.f90 2>&1 | FileCheck %s --check-prefix=COMPLEX
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/complex-array-part.f90 2>&1 | FileCheck %s --check-prefix=COMPLEX
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/complex-scalar-part.f90 2>&1 | FileCheck %s --check-prefix=COMPLEX
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/complex-scalar-part.f90 2>&1 | FileCheck %s --check-prefix=COMPLEX
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/complex-task.f90 2>&1 | FileCheck %s --check-prefix=IN-COMPLEX
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/complex-task.f90 2>&1 | FileCheck %s --check-prefix=IN-COMPLEX
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/complex-target.f90 2>&1 | FileCheck %s --check-prefix=IN-COMPLEX
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/complex-target.f90 2>&1 | FileCheck %s --check-prefix=IN-COMPLEX
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/complex-task-reduction.f90 2>&1 | FileCheck %s --check-prefix=TASK-COMPLEX
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/complex-task-reduction.f90 2>&1 | FileCheck %s --check-prefix=TASK-COMPLEX
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/complex-taskloop.f90 2>&1 | FileCheck %s --check-prefix=COMPLEX
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/complex-taskloop.f90 2>&1 | FileCheck %s --check-prefix=COMPLEX
! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/supported.f90 | FileCheck %s --check-prefix=SUPPORTED --implicit-check-not="not yet implemented"
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/supported.f90 | FileCheck %s --check-prefix=SUPPORTED --implicit-check-not="not yet implemented"

! All constructs that lower a REDUCTION clause call the shared reduction
! processor. Together with the existing taskloop and element tests, these cases
! form the complete construct-by-subobject matrix.

! ELEMENT: not yet implemented: REDUCTION of an array element using a user-defined reduction
! SECTION: not yet implemented: REDUCTION of a partial array section using a user-defined reduction
! IN-SECTION: not yet implemented: IN_REDUCTION of a partial array section using a user-defined reduction
! TASK-SECTION: not yet implemented: TASK_REDUCTION of a partial array section using a user-defined reduction
! COMPLEX: not yet implemented: REDUCTION of a complex part
! IN-COMPLEX: not yet implemented: IN_REDUCTION of a complex part
! TASK-COMPLEX: not yet implemented: TASK_REDUCTION of a complex part

! SUPPORTED-LABEL: func.func @_QPwhole_section
! SUPPORTED-LABEL: func.func @_QPpredefined_element
! SUPPORTED-LABEL: func.func @_QPpredefined_section

!--- parallel-section.f90
subroutine parallel_section(a)
  integer :: a(4)
  !$omp declare reduction(myred : integer : omp_out = omp_out + omp_in) &
  !$omp& initializer(omp_priv = 1)
  !$omp parallel reduction(myred : a(2:3))
  a(2:3) = a(2:3) + 1
  !$omp end parallel
end subroutine

!--- do-section.f90
subroutine do_section(a)
  integer :: a(4), i
  !$omp declare reduction(myred : integer : omp_out = omp_out + omp_in) &
  !$omp& initializer(omp_priv = 1)
  !$omp do reduction(myred : a(2:3))
  do i = 1, 1
    a(2:3) = a(2:3) + i
  end do
  !$omp end do
end subroutine

!--- sections-section.f90
subroutine sections_section(a)
  integer :: a(4)
  !$omp declare reduction(myred : integer : omp_out = omp_out + omp_in) &
  !$omp& initializer(omp_priv = 1)
  !$omp sections reduction(myred : a(2:3))
  !$omp section
  a(2:3) = a(2:3) + 1
  !$omp end sections
end subroutine

!--- scope-section.f90
subroutine scope_section(a)
  integer :: a(4)
  !$omp declare reduction(myred : integer : omp_out = omp_out + omp_in) &
  !$omp& initializer(omp_priv = 1)
  !$omp scope reduction(myred : a(2:3))
  a(2:3) = a(2:3) + 1
  !$omp end scope
end subroutine

!--- simd-element.f90
subroutine simd_element(a)
  integer :: a(4), i
  !$omp declare reduction(myred : integer : omp_out = omp_out + omp_in) &
  !$omp& initializer(omp_priv = 1)
  !$omp simd reduction(myred : a(2))
  do i = 1, 1
    a(2) = a(2) + i
  end do
end subroutine

!--- simd-section.f90
subroutine simd_section(a)
  integer :: a(4), i
  !$omp declare reduction(myred : integer : omp_out = omp_out + omp_in) &
  !$omp& initializer(omp_priv = 1)
  !$omp simd reduction(myred : a(2:3))
  do i = 1, 1
    a(2:3) = a(2:3) + i
  end do
end subroutine

!--- teams-element.f90
subroutine teams_element(a)
  integer :: a(4)
  !$omp declare reduction(myred : integer : omp_out = omp_out + omp_in) &
  !$omp& initializer(omp_priv = 1)
  !$omp teams reduction(myred : a(2))
  a(2) = a(2) + 1
  !$omp end teams
end subroutine

!--- teams-section.f90
subroutine teams_section(a)
  integer :: a(4)
  !$omp declare reduction(myred : integer : omp_out = omp_out + omp_in) &
  !$omp& initializer(omp_priv = 1)
  !$omp teams reduction(myred : a(2:3))
  a(2:3) = a(2:3) + 1
  !$omp end teams
end subroutine

!--- loop-element.f90
subroutine loop_element(a)
  integer :: a(4), i
  !$omp declare reduction(myred : integer : omp_out = omp_out + omp_in) &
  !$omp& initializer(omp_priv = 1)
  !$omp loop reduction(myred : a(2))
  do i = 1, 1
    a(2) = a(2) + i
  end do
  !$omp end loop
end subroutine

!--- loop-section.f90
subroutine loop_section(a)
  integer :: a(4), i
  !$omp declare reduction(myred : integer : omp_out = omp_out + omp_in) &
  !$omp& initializer(omp_priv = 1)
  !$omp loop reduction(myred : a(2:3))
  do i = 1, 1
    a(2:3) = a(2:3) + i
  end do
  !$omp end loop
end subroutine

!--- in-reduction-section.f90
subroutine in_reduction_section(a)
  integer :: a(4)
  !$omp declare reduction(+ : integer : omp_out = omp_out + omp_in) &
  !$omp& initializer(omp_priv = 1)
  !$omp task shared(a) in_reduction(+ : a(2:3))
  a(2:3) = a(2:3) + 1
  !$omp end task
end subroutine

!--- task-reduction-section.f90
subroutine task_reduction_section(a)
  integer :: a(4)
  !$omp declare reduction(+ : integer : omp_out = omp_out + omp_in) &
  !$omp& initializer(omp_priv = 1)
  !$omp taskgroup task_reduction(+ : a(2:3))
  a(2:3) = a(2:3) + 1
  !$omp end taskgroup
end subroutine

!--- complex-part.f90
subroutine complex_part(z)
  complex :: z(4)
  !$omp declare reduction(+ : real : omp_out = omp_out + omp_in) &
  !$omp& initializer(omp_priv = 1.0)
  !$omp parallel reduction(+ : z(2)%re)
  z(2)%re = z(2)%re + 1.0
  !$omp end parallel
end subroutine

!--- complex-array-part.f90
subroutine complex_array_part(z)
  complex :: z(4)
  !$omp parallel reduction(+ : z(2)%im)
  z(2)%im = z(2)%im + 1.0
  !$omp end parallel
end subroutine

!--- complex-scalar-part.f90
subroutine complex_scalar_part(z)
  complex :: z
  !$omp parallel reduction(+ : z%re)
  z%re = z%re + 1.0
  !$omp end parallel
end subroutine

!--- complex-task.f90
subroutine complex_task(z)
  complex :: z(4)
  !$omp task in_reduction(+ : z(2)%re)
  z(2)%re = z(2)%re + 1.0
  !$omp end task
end subroutine

!--- complex-target.f90
subroutine complex_target(z)
  complex :: z(4)
  !$omp target in_reduction(+ : z(2)%im) map(tofrom : z)
  z(2)%im = z(2)%im + 1.0
  !$omp end target
end subroutine

!--- complex-task-reduction.f90
subroutine complex_task_reduction(z)
  complex :: z
  !$omp taskgroup task_reduction(+ : z%im)
  z%im = z%im + 1.0
  !$omp end taskgroup
end subroutine

!--- complex-taskloop.f90
subroutine complex_taskloop(z)
  complex :: z(4)
  integer :: i
  !$omp taskloop reduction(+ : z(2)%re)
  do i = 1, 1
    z(2)%re = z(2)%re + i
  end do
end subroutine

!--- supported.f90
subroutine whole_section(a)
  integer :: a(4)
  !$omp declare reduction(myred : integer : omp_out = omp_out + omp_in) &
  !$omp& initializer(omp_priv = 1)
  !$omp parallel reduction(myred : a(:))
  a = a + 1
  !$omp end parallel
end subroutine

subroutine predefined_element(a)
  integer :: a(4)
  !$omp parallel reduction(+ : a(2))
  a(2) = a(2) + 1
  !$omp end parallel
end subroutine

subroutine predefined_section(a)
  integer :: a(4)
  !$omp parallel reduction(+ : a(2:3))
  a(2:3) = a(2:3) + 1
  !$omp end parallel
end subroutine
