! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck %s

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
  type(t1),pointer :: foo, foo2
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

! Test that file compiles ok and there is only one DICompositeType for "t1".
!CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "t1"{{.*}})
!CHECK-NOT: !DICompositeType(tag: DW_TAG_structure_type, name: "t1"{{.*}})
