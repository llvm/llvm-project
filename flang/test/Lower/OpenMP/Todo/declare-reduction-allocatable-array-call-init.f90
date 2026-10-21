! An allocatable-array reduction with a function-call initializer is not lowered
! (a single broadcast value can't call per element). See #186765.

! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: OpenMP user-defined reduction on an allocatable/pointer array

subroutine test_udr_array_call_init()
  interface
    integer function f()
    end function
  end interface
  integer, allocatable :: b(:)
  integer :: i
  !$omp declare reduction(myr : integer : omp_out = omp_out + omp_in) &
  !$omp&  initializer(omp_priv = f())
  allocate(b(4)); b = 0
  !$omp parallel do reduction(myr : b)
  do i = 1, 4
    b = b + i
  end do
end subroutine
