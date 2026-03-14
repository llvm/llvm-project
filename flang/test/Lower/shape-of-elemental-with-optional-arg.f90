! Test that the shape of the elemental call is properly
! computed as being rank 1, even though the only dummy
! argument is optional.

! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

subroutine test
  interface
     elemental function callee(arg1)
       integer, intent(in), optional :: arg1
       integer                       :: fun
     end function callee
  end interface
  integer :: arr(2)
  print *, callee(arr)
end subroutine test
! The PRINT statement must be lowered into a ranked print:
! CHECK: fir.call @_FortranAioOutputDescriptor
