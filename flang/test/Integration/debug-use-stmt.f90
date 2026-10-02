! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck %s

module testmod
  integer :: var_a = 10, var_b = 20, var_c = 30
  integer, parameter :: par_a = 11, par_b = 22
  integer, parameter :: tbl(4) = [1, 2, 3, 4]
end module testmod

module testmod2
  real :: var_x = 1.0, var_y = 2.0
end module testmod2

module testmod3
  real, parameter :: par_x = 3.5, par_y = 4.5
end module testmod3

program test_use
  use testmod, only: var_b, var_d => var_c, par_a, par_d => par_b, tbl
  use testmod2, var_z => var_y
  use testmod3, par_z => par_y
  implicit none
  print *, var_b
  print *, var_d
  print *, var_z
  print *, par_a, par_d, tbl(2), par_z
end program

! CHECK-DAG: [[TESTMOD:![0-9]+]] = !DIModule(scope: !{{.*}}, name: "testmod"
! CHECK-DAG: [[TESTMOD2:![0-9]+]] = !DIModule(scope: !{{.*}}, name: "testmod2"
! CHECK-DAG: [[TESTMOD3:![0-9]+]] = !DIModule(scope: !{{.*}}, name: "testmod3"

! CHECK-DAG: [[VAR_B:![0-9]+]] = distinct !DIGlobalVariable(name: "var_b", linkageName: "_QMtestmodEvar_b"
! CHECK-DAG: [[VAR_C:![0-9]+]] = distinct !DIGlobalVariable(name: "var_c", linkageName: "_QMtestmodEvar_c"
! CHECK-DAG: [[VAR_Y:![0-9]+]] = distinct !DIGlobalVariable(name: "var_y", linkageName: "_QMtestmod2Evar_y"
! CHECK-DAG: [[PAR_A:![0-9]+]] = distinct !DIGlobalVariable(name: "par_a", linkageName: "_QMtestmodECpar_a"
! CHECK-DAG: [[PAR_B:![0-9]+]] = distinct !DIGlobalVariable(name: "par_b", linkageName: "_QMtestmodECpar_b"
! CHECK-DAG: [[TBL:![0-9]+]] = distinct !DIGlobalVariable(name: "tbl", linkageName: "_QMtestmodECtbl"
! CHECK-DAG: [[PAR_Y:![0-9]+]] = distinct !DIGlobalVariable(name: "par_y", linkageName: "_QMtestmod3ECpar_y"

! CHECK-DAG: [[SP:![0-9]+]] = distinct !DISubprogram(name: "test_use", linkageName: "_QQmain"{{.*}}retainedNodes:

! Check that the full testmod module is not imported
! CHECK-NOT: !DIImportedEntity(tag: DW_TAG_imported_module, scope: [[SP]], entity: [[TESTMOD]]
! Check testmod imports: var_b directly (no rename), var_d as rename of var_c
! CHECK-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: [[SP]], entity: [[VAR_B]],{{.*}}file:{{.*}}line:
! CHECK-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "var_d", scope: [[SP]], entity: [[VAR_C]],{{.*}}file:{{.*}}line:
! A named constant reached through ONLY is imported the same way, whether it is
! scalar or an array, and whether or not it is renamed.
! CHECK-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: [[SP]], entity: [[PAR_A]],{{.*}}file:{{.*}}line:
! CHECK-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "par_d", scope: [[SP]], entity: [[PAR_B]],{{.*}}file:{{.*}}line:
! CHECK-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: [[SP]], entity: [[TBL]],{{.*}}file:{{.*}}line:

! Check testmod2 import: module imported with rename in elements array
! The module import should have elements containing the var_z rename
! CHECK-DAG: [[MOD2_IMPORT:![0-9]+]] = !DIImportedEntity(tag: DW_TAG_imported_module, scope: [[SP]], entity: [[TESTMOD2]],{{.*}}elements: [[ELEMENTS:![0-9]+]]
! CHECK-DAG: [[ELEMENTS]] = !{[[VAR_Z:![0-9]+]]}
! CHECK-DAG: [[VAR_Z]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "var_z",{{.*}}entity: [[VAR_Y]],

! A named constant renamed without ONLY lands in the same elements array.
! CHECK-DAG: [[MOD3_IMPORT:![0-9]+]] = !DIImportedEntity(tag: DW_TAG_imported_module, scope: [[SP]], entity: [[TESTMOD3]],{{.*}}elements: [[ELEMENTS3:![0-9]+]]
! CHECK-DAG: [[ELEMENTS3]] = !{[[PAR_Z:![0-9]+]]}
! CHECK-DAG: [[PAR_Z]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "par_z",{{.*}}entity: [[PAR_Y]],
