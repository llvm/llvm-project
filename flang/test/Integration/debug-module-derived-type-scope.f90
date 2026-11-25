! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck %s

! Test that a derived type defined in a module gets the scope of that module.

module inner_m
  type :: leaf
    integer :: v
  end type
end module

module outer_m
  use inner_m
  type :: node
    type(leaf) :: l
  end type
  interface
    module subroutine smod_sub()
    end subroutine
  end interface
contains
  subroutine msub()
    type :: local_t
      integer :: w
    end type
    type(local_t) :: lv
    lv%w = 1
    print *, lv%w
  end subroutine
end module

submodule (outer_m) impl
  type :: sub_t
    integer :: s
  end type
contains
  module subroutine smod_sub()
    type(sub_t) :: sv
    sv%s = 3
    print *, sv%s
  end subroutine
end submodule

subroutine first()
  use outer_m
  type(node) :: n1
  n1%l%v = 1
  print *, n1%l%v
end subroutine

! CHECK-DAG: ![[INNER:[0-9]+]] = !DIModule({{.*}}name: "inner_m"
! CHECK-DAG: ![[OUTER:[0-9]+]] = !DIModule({{.*}}name: "outer_m"
! CHECK-DAG: ![[IMPL:[0-9]+]] = !DIModule({{.*}}name: "outer_m.impl"
! CHECK-DAG: ![[MSUB:[0-9]+]] = distinct !DISubprogram(name: "msub"
! CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "leaf", scope: ![[INNER]],
! CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "node", scope: ![[OUTER]],
! CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "sub_t", scope: ![[IMPL]],
! CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "local_t", scope: ![[MSUB]],
