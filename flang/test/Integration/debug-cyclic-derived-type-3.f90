! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o -

! mainly test that this program does not cause an assertion failure
! testcase for issue 122024

module m1
  type t1
    type(t2),pointer :: x1
  end type
  type t2
    type(t3),pointer :: x2
  end type
  type t3
    type(t1),pointer :: x3
  end type
end

program test
  use m1
  type(t1),pointer :: foo
  allocate(foo)
  allocate(foo%x1)
  allocate(foo%x1%x2)
  allocate(foo%x1%x2%x3)
  call sub1(foo%x1)
  print *,'done'
end program

subroutine sub1(bar)
  use m1
  type(t2) :: bar
end subroutine
