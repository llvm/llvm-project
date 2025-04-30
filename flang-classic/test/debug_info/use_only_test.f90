! REQUIRES: llvm-17

!RUN: %flang -g -S -emit-llvm %s_mod.f90 %s
!RUN: cat use_only_test.ll | FileCheck %s

!CHECK: [[MOD_VAR:![0-9]+]] = distinct !DIGlobalVariable(name: "mod_var1"
!CHECK: distinct !DICompileUnit
!CHECK-SAME: imports: [[IMPLIST:![0-9]+]]
!CHECK: [[IMPLIST]] = !{}
!CHECK: !DISubprogram(name: "main"
!CHECK-SAME: retainedNodes: [[RETAINED:![0-9]+]]
!CHECK: [[RETAINED]] = !{[[RETAIN1:![0-9]+]]}
!CHECK: [[RETAIN1]] = !DIImportedEntity(tag: DW_TAG_imported_declaration
!CHECK-SAME: entity: [[MOD_VAR]]
!CHECK-NOT: !DIImportedEntity(tag: DW_TAG_imported_module

program main
  use use_only_mod, only: mod_var1
  implicit none
  print*, mod_var1
end
