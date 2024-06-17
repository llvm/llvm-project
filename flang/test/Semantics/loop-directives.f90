! RUN: %python %S/test_errors.py %s %flang

subroutine empty
  ! ERROR: A DO loop must follow the VECTOR ALWAYS directive
  !dir$ vector always
end subroutine empty

subroutine non_do
  ! ERROR: A DO loop must follow the VECTOR ALWAYS directive
  !dir$ vector always
  a = 1
end subroutine non_do

subroutine execution_part
  do i=1,10
  ! ERROR: A DO loop must follow the VECTOR ALWAYS directive
  !dir$ vector always
  end do
end subroutine execution_part
