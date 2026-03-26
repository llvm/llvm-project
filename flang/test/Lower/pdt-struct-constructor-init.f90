! RUN: not bbc -emit-hlfir %s 2>&1 | FileCheck %s

program main
  type ty0(k)
    integer,kind::k
    integer :: ii
  end type

  type ty(k,l)
    integer,kind::k
    integer,len ::l
    type(ty0(2)) :: cmp(1) = [ty0(2)(4)]
  end type

  type(ty(2,4)) :: obj
end program

! CHECK: not yet implemented: parameterized derived types
