! RUN: flang -fc1 -emit-hlfir %s -o - | FileCheck %s

program main
  call test06()
  print *,'pass'
end program main

subroutine test06()
  type ty1
     integer ,allocatable :: a(:,:,:)
  end type ty1
  type(ty1) :: str(1)
  integer ,allocatable :: b(:,:,:)
  allocate(str(1)%a(1,1,1),b(1,1,1))
  b=1
  write(6,*) "b                                 = ", b
  write(6,*) "reshape((/(b,jj=1,1)/),(/1,1,1/)) = ", reshape((/(b,jj=1,1)/),(/1,1,1/))
  where ((/.true./)) str=(/(ty1(reshape((/(b,jj=1,1)/),(/1,1,1/))),ii=1,1)/)
  ! CHECK: hlfir.exactly_once : !hlfir.expr<1x!fir.type<_QFtest06Tty1{a:!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>}>>
end subroutine test06
