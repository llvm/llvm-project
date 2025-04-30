! REQUIRES: llvm-17

!RUN: %flang -g -S -emit-llvm %s -o - | FileCheck %s


!CHECK-DAG: [[VAR1:![0-9]+]] = distinct !DIGlobalVariable(name: "var1"
!CHECK-DAG: [[MYMOD:![0-9]+]] = !DIModule(scope: {{![0-9]+}}, name: "mymod"
!CHECK-DAG: [[VAR2:![0-9]+]] = distinct !DIGlobalVariable(name: "var2"
!CHECK-DAG: [[VAR3:![0-9]+]] = distinct !DIGlobalVariable(name: "var3"

!CHECK-DAG: !DIImportedEntity(tag: DW_TAG_imported_module, scope: [[USE_ALL:![0-9]+]], entity: [[MYMOD]]
!CHECK-DAG: [[USE_ALL]] = distinct !DISubprogram(name: "use_all"

!CHECK-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: [[USE_RESTRICTED:![0-9]+]], entity: [[VAR1]]
!CHECK-DAG: [[USE_RESTRICTED]] = distinct !DISubprogram(name: "use_restricted"

!CHECK: [[USE_RENAMED:![0-9]+]] = distinct !DISubprogram(name: "use_renamed"
!CHECK-SAME: retainedNodes: [[RETAINED:![0-9]+]]
!CHECK: [[RETAINED]] = !{[[RETAIN1:![0-9]+]]
!CHECK: [[RETAIN1]] = !DIImportedEntity(tag: DW_TAG_imported_module, scope: [[USE_RENAMED]], entity: [[MYMOD]]
!CHECK-SAME: elements: [[RENAMES:![0-9]+]]
!CHECK: [[RENAMES]] = !{[[RENAME1:![0-9]+]]}
!CHECK: [[RENAME1]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "var4", scope: [[USE_RENAMED:![0-9]+]], entity: [[VAR1]]

!CHECK-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "var4", scope: [[USE_RESTRICTED_RENAMED:![0-9]+]], entity: [[VAR1]]
!CHECK-DAG: [[USE_RESTRICTED_RENAMED]] = distinct !DISubprogram(name: "use_restricted_renamed"

module mymod
  integer :: var1 = 11
  integer :: var2 = 12
  integer :: var3 = 13
end module mymod

Program main
  call use_all()
  call use_restricted()
  call use_renamed()
  call use_restricted_renamed()
  contains
    subroutine use_all()
      use mymod
      print *, var1
    end subroutine use_all
    subroutine use_restricted()
      use mymod, ONLY: var1
      print *, var1
    end subroutine use_restricted
    subroutine use_renamed()
      use mymod, var4 => var1
      print *, var4
    end subroutine use_renamed
    subroutine use_restricted_renamed()
      use mymod, ONLY: var4 => var1
      print *, var4
    end subroutine use_restricted_renamed
end program main
