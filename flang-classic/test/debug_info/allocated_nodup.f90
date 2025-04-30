!RUN: %flang -gdwarf-4 -S -emit-llvm %s -o - | FileCheck %s

!CHECK-DAG: distinct !DIGlobalVariable(name: "arr",{{.*}}type: [[TYPE:![0-9]+]]
!CHECK-DAG: [[TYPE]] = !DICompositeType(tag: DW_TAG_array_type, baseType: [[DTYPE:![0-9]+]]
!CHECK-DAG: [[DTYPE]] = !DICompositeType(tag: DW_TAG_structure_type, name: "dtype"{{.*}}elements: [[MEMBERS:![0-9]+]]
!CHECK-DAG: [[MEMBERS]] = !{[[MEM1:![0-9]+]]
!CHECK-DAG: [[MEM1]] = !DIDerivedType(tag: DW_TAG_member, name: "memfunptr",{{.*}}baseType: [[FUNPTRTYPE:![0-9]+]]
!CHECK-DAG: [[FUNPTRTYPE]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: [[FUNTYPE:![0-9]+]]
!CHECK-DAG: [[FUNTYPE]] = !DISubroutineType(types: [[FUNSIGNATURE:![0-9]+]])
!CHECK-DAG: [[FUNSIGNATURE]] = !{[[DTYPE]]

module pdt
  type dtype
    procedure (func), pointer, nopass :: memfunptr
    integer, allocatable :: memalcarr(:)
  end type dtype
contains
  function func()
    class (dtype), allocatable :: func
  end function func
end module pdt

program main
  use pdt
  type (dtype) arr(3)
  allocate(arr(1)%memalcarr(10))
  arr(1)%memalcarr=9
  print *, arr(1)%memalcarr
end program main
