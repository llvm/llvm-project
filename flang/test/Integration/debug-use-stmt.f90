! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck %s

module testmod
  integer :: var_a = 10, var_b = 20, var_c = 30
end module testmod

module testmod2
  real :: var_x = 1.0, var_y = 2.0
end module testmod2

program test_use
  use testmod, only: var_b, var_d => var_c
  use testmod2, var_z => var_y
  implicit none
  print *, var_b
  print *, var_d
  print *, var_z
end program

! CHECK-DAG: [[TESTMOD:![0-9]+]] = !DIModule(scope: !{{.*}}, name: "testmod"
! CHECK-DAG: [[TESTMOD2:![0-9]+]] = !DIModule(scope: !{{.*}}, name: "testmod2"

! CHECK-DAG: [[VAR_B:![0-9]+]] = distinct !DIGlobalVariable(name: "var_b", linkageName: "_QMtestmodEvar_b"
! CHECK-DAG: [[VAR_C:![0-9]+]] = distinct !DIGlobalVariable(name: "var_c", linkageName: "_QMtestmodEvar_c"
! CHECK-DAG: [[VAR_Y:![0-9]+]] = distinct !DIGlobalVariable(name: "var_y", linkageName: "_QMtestmod2Evar_y"

! CHECK-DAG: [[SP:![0-9]+]] = distinct !DISubprogram(name: "TEST_USE", linkageName: "_QQmain"{{.*}}retainedNodes:

! Check testmod imports: var_b directly (no rename), var_d as rename of var_c
! CHECK-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: [[SP]], entity: [[VAR_B]],{{.*}}file:{{.*}}line:
! CHECK-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "var_d", scope: [[SP]], entity: [[VAR_C]],{{.*}}file:{{.*}}line:

! Check testmod2 import: module imported with rename in elements array
! The module import should have elements containing the var_z rename
! CHECK-DAG: [[MOD2_IMPORT:![0-9]+]] = !DIImportedEntity(tag: DW_TAG_imported_module, scope: [[SP]], entity: [[TESTMOD2]],{{.*}}elements: [[ELEMENTS:![0-9]+]]
! CHECK-DAG: [[ELEMENTS]] = !{[[VAR_Z:![0-9]+]]}
! CHECK-DAG: [[VAR_Z]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "var_z",{{.*}}entity: [[VAR_Y]],

