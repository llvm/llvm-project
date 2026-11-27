! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck %s

module mod
    type cart
        integer :: x1
    end type
contains
    type(cart) function build(x)
        integer :: x
        build%x1 = x
    end function
end module

program test
    use mod
    implicit none
    type(cart) :: b
    b = build(4)
    print *, b
end program

! CHECK-DAG: ![[CART:[0-9]+]] = {{.*}}!DICompositeType(tag: DW_TAG_structure_type, name: "cart"{{.*}})
! CHECK-DAG: ![[INT:[0-9]+]] = !DIBasicType(name: "integer(kind=4)"{{.*}})
! CHECK-DAG: ![[TYPES:[0-9]+]] = !{![[CART]], ![[INT]]}
! CHECK-DAG: ![[SUBR_TYPE:[0-9]+]] = !DISubroutineType({{.*}}types: ![[TYPES]])
! CHECK-DAG: distinct !DISubprogram(name: "build"{{.*}}type: ![[SUBR_TYPE]]{{.*}})
