! RUN: %python %S/../test_errors.py %s %flang -fopenacc

subroutine compute()
  integer :: a(3), c, i

  a = 1
  !ERROR: 'c' appears in more than one data-sharing clause on the same OpenACC directive
  !$acc parallel firstprivate(c) private(c)
  do i = 1, 3
    a(i) = c
  end do
  !$acc end parallel
end subroutine compute

program mm
  call compute()
end
