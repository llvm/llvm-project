! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck %s

! Each instance of a parameterized type is named after its kind values, as
! gfortran does, so that instances used side by side can be told apart. A kind
! value left to its default is part of the name too. A negative value is
! spelled with `N`, as a debugger cannot look up a name with a minus sign.

module m
  type :: pdt(k)
    integer, kind :: k
    integer(k) :: a
  end type
  type :: pdt2(k1, k2)
    integer, kind :: k1
    integer, kind :: k2 = 2
    real(k1) :: x
    integer(k2) :: n(4)
  end type
  type :: pdt3(k)
    integer, kind :: k
    integer :: a(abs(k))
  end type
end module

subroutine sub()
  use m
  type(pdt(4)) :: p4
  type(pdt(8)) :: p8
  type(pdt2(8)) :: q
  type(pdt3(-2)) :: r
  p4%a = 4
  p8%a = 8
  q%x = 1.0
  q%n = 2
  r%a = 3
  print *, p4%a, p8%a, q%x, q%n, r%a
end subroutine

! CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "PDTpdt_4",{{.*}} size: 32,
! CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "PDTpdt_8",{{.*}} size: 64,
! CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "PDTpdt2_8_2",{{.*}} size: 128,
! CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "PDTpdt3_N2",{{.*}} size: 64,
