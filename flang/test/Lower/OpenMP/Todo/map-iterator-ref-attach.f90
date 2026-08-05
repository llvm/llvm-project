! RUN: split-file %s %t
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=61 -o - %t/ref-ptr.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=61 -o - %t/ref-ptee.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=61 -o - %t/ref-ptr-ptee.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=61 -o - %t/attach-always.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=61 -o - %t/attach-never.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=61 -o - %t/attach-auto.f90 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=61 \
! RUN:   -o /dev/null %t/plain-object.f90

! CHECK: not yet implemented: iterator modifier with reference or attach
! CHECK-SAME: modifier

!--- ref-ptr.f90
subroutine ref_ptr(a)
  integer, pointer :: a(:)
  integer :: i

  !$omp target data map(ref_ptr, iterator(i = 1:10), to: a(i))
  !$omp end target data
end subroutine

!--- ref-ptee.f90
subroutine ref_ptee(a)
  integer, pointer :: a(:)
  integer :: i

  !$omp target data map(ref_ptee, iterator(i = 1:10), to: a(i))
  !$omp end target data
end subroutine

!--- ref-ptr-ptee.f90
subroutine ref_ptr_ptee(a)
  integer, pointer :: a(:)
  integer :: i

  !$omp target data map(ref_ptr_ptee, iterator(i = 1:10), to: a(i))
  !$omp end target data
end subroutine

!--- attach-always.f90
subroutine attach_always(a)
  integer, pointer :: a(:)
  integer :: i

  !$omp target data map(attach(always), iterator(i = 1:10), to: a(i))
  !$omp end target data
end subroutine

!--- attach-never.f90
subroutine attach_never(a)
  integer, pointer :: a(:)
  integer :: i

  !$omp target data map(attach(never), iterator(i = 1:10), to: a(i))
  !$omp end target data
end subroutine

!--- attach-auto.f90
subroutine attach_auto(a)
  integer, pointer :: a(:)
  integer :: i

  !$omp target data map(attach(auto), iterator(i = 1:10), to: a(i))
  !$omp end target data
end subroutine

!--- plain-object.f90
subroutine plain_object(a)
  integer, pointer :: a(:)
  integer :: i

  !$omp target data map(ref_ptr, iterator(i = 1:10), to: a)
  !$omp end target data
end subroutine
