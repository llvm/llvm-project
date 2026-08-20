! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck %s

! The entities a USE statement imports into a scope are emitted in source order,
! which keeps the debug information, and with it the object file, identical from
! one run of the compiler to the next.

module m1
  integer :: a1 = 1, b1 = 2, c1 = 3
end module m1

module m2
  integer :: a2 = 4
end module m2

module m3
  integer :: a3 = 5, b3 = 6
end module m3

program test_order
  use m1, only: a1, b1, z1 => c1
  use m2
  use m3, x3 => a3, y3 => b3
  implicit none
  print *, a1, b1, z1, a2, x3, y3
end program

! CHECK-DAG: [[A1:![0-9]+]] = distinct !DIGlobalVariable(name: "a1"
! CHECK-DAG: [[B1:![0-9]+]] = distinct !DIGlobalVariable(name: "b1"
! CHECK-DAG: [[C1:![0-9]+]] = distinct !DIGlobalVariable(name: "c1"
! CHECK-DAG: [[M2:![0-9]+]] = !DIModule(scope: !{{.*}}, name: "m2"
! CHECK-DAG: [[M3:![0-9]+]] = !DIModule(scope: !{{.*}}, name: "m3"
! CHECK-DAG: [[A3:![0-9]+]] = distinct !DIGlobalVariable(name: "a3"
! CHECK-DAG: [[B3:![0-9]+]] = distinct !DIGlobalVariable(name: "b3"
! CHECK-DAG: [[SP:![0-9]+]] = distinct !DISubprogram(name: "TEST_ORDER"{{.*}}retainedNodes: [[NODES:![0-9]+]]

! CHECK: [[NODES]] = !{[[E1:![0-9]+]], [[E2:![0-9]+]], [[E3:![0-9]+]], [[E4:![0-9]+]], [[E5:![0-9]+]]}
! CHECK-NEXT: [[E1]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: [[SP]], entity: [[A1]],
! CHECK-NEXT: [[E2]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: [[SP]], entity: [[B1]],
! CHECK-NEXT: [[E3]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "z1", scope: [[SP]], entity: [[C1]],
! CHECK-NEXT: [[E4]] = !DIImportedEntity(tag: DW_TAG_imported_module, scope: [[SP]], entity: [[M2]],

! The renames a USE statement without ONLY brings in are the children of the
! module import, and they keep source order too.
! CHECK-NEXT: [[E5]] = !DIImportedEntity(tag: DW_TAG_imported_module, scope: [[SP]], entity: [[M3]],{{.*}}elements: [[ELEMENTS:![0-9]+]]
! CHECK-NEXT: [[ELEMENTS]] = !{[[C1E:![0-9]+]], [[C2E:![0-9]+]]}
! CHECK-NEXT: [[C1E]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "x3", scope: [[SP]], entity: [[A3]],
! CHECK-NEXT: [[C2E]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "y3", scope: [[SP]], entity: [[B3]],
