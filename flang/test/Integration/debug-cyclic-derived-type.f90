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

! CHECK-DAG: ![[T1:[0-9]+]] = {{.*}}!DICompositeType(tag: DW_TAG_structure_type, name: "t1"{{.*}}elements: ![[T1_ELEMS:[0-9]+]])
! CHECK-DAG: ![[T1_ELEMS]] = !{![[T1_ELEM1:[0-9]+]]}
! CHECK-DAG: ![[T1_ELEM1]] = !DIDerivedType(tag: DW_TAG_member, name: "p", baseType: ![[T2P:[0-9]+]]{{.*}})
! CHECK-DAG: ![[T2P]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[T2:[0-9]+]]{{.*}})

! CHECK-DAG: ![[T2]] = {{.*}}!DICompositeType(tag: DW_TAG_structure_type, name: "t2"{{.*}}elements: ![[T2_ELEMS:[0-9]+]])
! CHECK-DAG: ![[T2_ELEMS]] = !{![[T2_ELEM1:[0-9]+]]}
! CHECK-DAG: ![[T2_ELEM1]] = !DIDerivedType(tag: DW_TAG_member, name: "v1", baseType: ![[T1]]{{.*}})
