! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Checks that no fir store is present with fir.class<none> as the first operand.
! Regression test for bug: FIR lowering failure on polymorphic assignment.

! CHECK-NOT: fir.store{{.*}}!fir.class<none>
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
end subroutine s1

program main
  call s1
  print *,'pass'
end program main
