! RUN: %python %S/test_errors.py %s %flang_fc1
subroutine s1()
  character(10) str
  character(10) str1
  !ERROR: Cannot reference function 'str' as data
  print *, str(1:9), str(7)
  block
    character(10) str2
    character(10) str3
    !ERROR: Cannot reference function 'str1' as data
    print *, str1(1:9), str1(7)
    print *, str2(1:9) ! substring is ok
    !ERROR: 'str2' is not a callable procedure
    print *, str2(7)
    !ERROR: Cannot reference function 'str3' as data
    print *, str3(7), str3(1:9)
  end block
end subroutine s1

subroutine s2()
  character(10) func
  !ERROR: Cannot reference function 'func' as data
  print *, func(7), func(1:9)
end subroutine s2

subroutine s3()
  real(8) :: func
  !ERROR: Cannot reference function 'func' as data
  print *, func(7), func(1:6)
end subroutine s3

subroutine s4()
  real(8) :: local
  real(8) :: local1
  !ERROR: Cannot reference function 'local' as data
  print *, local(1:6), local(7)
  !ERROR: Cannot reference function 'local1' as data
  print *, local1(7), local1(1:6)
end subroutine s4

subroutine s5(arg)
  integer :: iVar
  external :: arg
  iVar = loc(arg)
end subroutine s5
