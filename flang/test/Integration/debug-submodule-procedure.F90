! RUN: rm -rf %t && mkdir -p %t
! RUN: %flang_fc1 -fsyntax-only -DSTEP=1 -J%t %s
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone -J%t %s -o - \
! RUN:   | FileCheck %s

#if STEP == 1
module subpar
  implicit none
  interface
    module subroutine hello()
    end subroutine
  end interface
end module subpar
#else
submodule (subpar) subkid
contains
  module subroutine hello()
    print *, 'hello from submodule'
  end subroutine hello
end submodule subkid
#endif

! CHECK-DAG: ![[MOD:[0-9]+]] = !DIModule(scope: ![[#]], name: "subpar.subkid"
! CHECK-DAG: !DISubprogram(name: "hello", linkageName: "_QMsubparPhello", scope: ![[MOD]]

! The submodule imports its ancestor.
! CHECK-DAG: ![[ANC:[0-9]+]] = !DIModule(scope: null, name: "subpar", isDecl: true)
! CHECK-DAG: !DIImportedEntity(tag: DW_TAG_imported_module, scope: ![[MOD]], entity: ![[ANC]]
