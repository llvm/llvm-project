! RUN: %not_todo_cmd bbc -emit-fir %s -o -  2>&1 | FileCheck %s

interface
  pure function chfunc(a,b)
    character(*),intent(in) :: a,b
    character(3) :: chfunc
  end function
  end interface
  character(3) x(5)
  print*, reduce(x,chfunc)
end program

! CHECK: not yet implemented: intrinsic: reduce
