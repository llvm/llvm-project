! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck  %s

module m
 type t1
   type(t2), pointer :: p
 end type
 type t2
   type(t1) :: v1
 end type
 type(t1) :: v2
 type(t2) :: v3
end module

! CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "t1"{{.*}})
! CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "t2"{{.*}})
