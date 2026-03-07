! RUN: %python %S/test_errors.py %s %flang_fc1

Module m1
  external::sub
  type ty
    procedure(),pointer,nopass :: ptr5=>sub
  end type
  procedure(),pointer:: ptr6=>sub
end module
 
use m1
  integer::jj =4
  call ptr6(10)
  print*,"Pass"
end
 
subroutine sub(a)
  integer::a
  print*,"sub"
end subroutine