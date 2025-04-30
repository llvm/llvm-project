!RUN: %flang -gdwarf-4 -S -emit-llvm %s -o - | FileCheck %s

!CHECK: distinct !DIGlobalVariable(name: "dvar",
!CHECK-SAME: type: [[TYPE:![0-9]+]]
!CHECK: [[TYPE]] = !DICompositeType(tag: DW_TAG_structure_type, name: "dtype"
!CHECK-SAME: elements: [[MEMBERS:![0-9]+]]
!CHECK: [[MEMBERS]] = !{[[MEM1:![0-9]+]], [[MEM2:![0-9]+]], [[MEM3:![0-9]+]], [[MEM4:![0-9]+]], [[MEM5:![0-9]+]]
!CHECK: [[MEM1]] = !DIDerivedType(tag: DW_TAG_member, name: "i",
!CHECK-SAME: baseType: [[INTTYP:![0-9]+]]
!CHECK: [[MEM2]] = !DIDerivedType(tag: DW_TAG_member, name: "sclrptr",
!CHECK-SAME: baseType: [[SCLRPTRTYP:![0-9]+]]
!CHECK: [[SCLRPTRTYP]] = !DIDerivedType(tag: DW_TAG_pointer_type,
!CHECK-SAME: baseType: [[REALTYP:![0-9]+]]
!CHECK: [[REALTYP]] = !DIBasicType(name: "real"
!CHECK: [[MEM3]] = !DIDerivedType(tag: DW_TAG_member, name: "arrptr",
!CHECK-SAME: baseType: [[ARRTYP:![0-9]+]]
!CHECK: [[ARRTYP]] = !DICompositeType(tag: DW_TAG_array_type,
!CHECK: [[MEM4]] = !DIDerivedType(tag: DW_TAG_member, name: "dtptr",
!CHECK-SAME: baseType: [[DTTYP:![0-9]+]]
!CHECK: [[DTTYP]] = !DIDerivedType(tag: DW_TAG_pointer_type,
!CHECK: [[MEM5]] = !DIDerivedType(tag: DW_TAG_member, name: "dtarrptr",
!CHECK-SAME: baseType: [[DTARRTYP:![0-9]+]]
!CHECK: [[DTARRTYP]] = !DICompositeType(tag: DW_TAG_array_type,

program main
  implicit none

  type dtyp1
    integer :: scalar
    integer :: arr(10)
  end type dtyp1

  type dtype
    integer :: i
    real, pointer :: sclrptr
    integer, pointer :: arrptr(:)
    type(dtyp1), pointer :: dtptr
    type(dtyp1), pointer :: dtarrptr(:)
  end type dtype

  real, target :: rval
  integer, target :: arr(10) = (/0,2,4,6,8,1,3,5,7,9/)
  type(dtyp1), target :: dtvar
  type(dtyp1), target :: dtarr(10)
  type(dtype) :: dvar

  dtvar%scalar = 5
  dtvar%arr = 99
  dtarr = dtvar
  dtarr(5)%scalar = 55
  dvar%i = 4
  dvar%sclrptr => rval
  dvar%arrptr => arr
  dvar%dtptr => dtvar
  dvar%dtarrptr => dtarr

end program main
