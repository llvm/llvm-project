! RUN: flang -fc1 -emit-hlfir %s -o - | FileCheck %s

module m1
  type x
  end type x
  logical,parameter::t=.true.,f=.false.
  logical::mask(3)=[t,f,t]
end module m1

subroutine s1
  use m1
  class(*),allocatable::v(:),u(:)
  allocate(x::v(3))
  allocate(x::u(3))
  where(mask)
     u=v
  end where
! CHECK: hlfir.region_assign
! CHECK: !fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>>>
end subroutine s1

program main
  call s1
  print *,'pass'
end program main
