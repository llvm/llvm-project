! RUN: rm -rf %t && mkdir -p %t
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone -J%t %s -o - \
! RUN:   | FileCheck %s

! Test the import chain of a nested submodule: it imports the submodule that
! contains it, which imports the module at the root.

module nested_shapes
  implicit none
  integer :: root_var = 1
  interface
    module subroutine go()
    end subroutine
  end interface
end module nested_shapes

submodule (nested_shapes) mid
  integer :: mid_var = 2
end submodule mid

submodule (nested_shapes:mid) leaf
  integer :: leaf_var = 3
contains
  module subroutine go()
    print *, root_var, mid_var, leaf_var
  end subroutine go
end submodule leaf

! CHECK-DAG: ![[ROOT:[0-9]+]] = !DIModule(scope: ![[#]], name: "nested_shapes"
! CHECK-DAG: ![[MID:[0-9]+]] = !DIModule(scope: ![[#]], name: "nested_shapes.mid"
! CHECK-DAG: ![[LEAF:[0-9]+]] = !DIModule(scope: ![[#]], name: "nested_shapes.leaf"

! CHECK-DAG: !DIImportedEntity(tag: DW_TAG_imported_module, scope: ![[MID]], entity: ![[ROOT]]
! CHECK-DAG: !DIImportedEntity(tag: DW_TAG_imported_module, scope: ![[LEAF]], entity: ![[MID]]

! An entity stays in the submodule that declares it.
! CHECK-DAG: !DIGlobalVariable(name: "root_var", {{.*}}scope: ![[ROOT]],
! CHECK-DAG: !DIGlobalVariable(name: "mid_var", {{.*}}scope: ![[MID]],
! CHECK-DAG: !DIGlobalVariable(name: "leaf_var", {{.*}}scope: ![[LEAF]],
! CHECK-DAG: !DISubprogram(name: "go", {{.*}}scope: ![[LEAF]],
