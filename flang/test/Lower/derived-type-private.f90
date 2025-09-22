! Test lowering of derived type with private attribute
! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

program main
  call test02()
  print *,"pass"
end program main

module mod2
  type,private:: tt
     integer :: ip = 1
  end type tt
  type,extends(tt):: ty1
  ! CHECK: fir.global @_QMmod2Estr : !fir.type<_QMmod2Tty1{_QMmod2Tty1.tt:!fir.type<_QMmod2Ttt{ip:i32}>,i1:i32,i1p:!fir.type<_QMmod2Ttt{ip:i32}>,i1a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>
     integer :: i1 = 1
     type(tt) :: i1p = tt(2)
     integer,allocatable :: i1a(:)
  end type ty1
  type(ty1) :: str
end module mod2

subroutine test02()
  use mod2
  integer,allocatable :: ia(:)
  allocate(ia(10))
  ia=2
  str=ty1(i1a=ia)
  if (str%i1.ne.1) print *,'ng'
end subroutine test02
