! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck  %s

! mainly test that this program does not cause an assertion failure
module m
 type t2
   type(t1), pointer :: p1
 end type
 type t1
   type(t2), pointer :: p2
   integer abc
 end type
 type(t1) :: tee1
end module

program test
  use m
  type(t2) :: lc2
  print *, lc2%p1%abc
end program test

! CHECK-DAG: DICompositeType(tag: DW_TAG_structure_type, name: "t1"{{.*}})
! CHECK-DAG: DICompositeType(tag: DW_TAG_structure_type, name: "t2"{{.*}})
