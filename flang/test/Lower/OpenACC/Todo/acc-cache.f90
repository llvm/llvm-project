! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %s -o - 2>&1 | FileCheck %s

! CHECK: not yet implemented: OpenACC cache directive

subroutine test_cache()
  integer, parameter :: n = 10
  real, dimension(n) :: a, b
  integer :: i

  !$acc loop
  do i = 1, n
    !$acc cache(b)
    a(i) = b(i)
  end do
end subroutine
