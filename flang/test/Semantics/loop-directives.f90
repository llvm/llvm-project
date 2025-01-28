! RUN: %python %S/test_errors.py %s %flang_fc1 -Werror
! RUN: %python %S/test_errors.py %s %flang_fc1 -fopenacc -Werror

subroutine empty
  ! WARNING: A DO loop must follow the VECTOR ALWAYS directive
  !dir$ vector always
end subroutine empty

subroutine non_do
  ! WARNING: A DO loop must follow the VECTOR ALWAYS directive
  !dir$ vector always
  a = 1
end subroutine non_do

subroutine execution_part
  do i=1,10
  ! WARNING: A DO loop must follow the VECTOR ALWAYS directive
  !dir$ vector always
  end do
end subroutine execution_part

! OK
subroutine test_vector_always_before_acc(a, b, c)
  real, dimension(10) :: a,b,c
  !dir$ vector always
  !$acc loop
  do i=1,N
    a(i) = b(i) + c(i)
  enddo
end subroutine
