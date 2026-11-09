! RUN: not %flang_fc1 -emit-hlfir -fcoarray %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: coarray: initialization of module or submodule variable. 

module m
  integer :: x[*] = 5
end module

program m
  use m
  print *, x
end program

