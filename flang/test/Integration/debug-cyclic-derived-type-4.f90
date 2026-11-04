! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck  %s

! Same as debug-cyclic-derived-type-2.f90 but using class instead of type.
module m
 type t2
   class(t1), pointer :: p1
 end type
 type t1
   class(t2), pointer :: p2
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
