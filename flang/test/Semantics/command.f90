! RUN: %python %S/test_errors.py %s %flang_fc1
program command
  implicit none
  Integer(1) :: i1
  Integer(2) :: i2
  Integer(4) :: i4
  Integer(8) :: i8
  Integer(16) :: i16
	Integer :: a
  !ERROR: Actual argument for 'length=' has bad type or kind 'INTEGER(1)'
  call get_command(length=i1)
  !OK:
  call get_command(length=i2)
  !OK:
  call get_command(length=i4)
  !OK:
  call get_command(length=i8)
  !OK:
  call get_command(length=i16)
  !ERROR: Actual argument for 'length=' has bad type or kind 'INTEGER(1)'
  call get_command_argument(number=a,length=i1)
  !OK:
  call get_command_argument(number=a,length=i2)
  !OK:
  call get_command_argument(number=a,length=i4)
  !OK:
  call get_command_argument(number=a,length=i8)
  !OK:
  call get_command_argument(number=a,length=i16)
end program