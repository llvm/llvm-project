!RUN: %flang -g -S -emit-llvm %s -o - | FileCheck %s

!CHECK: !DIDerivedType(tag: DW_TAG_member, name: "proc1"
!CHECK-SAME: baseType: [[SPTRTYPE:![0-9]+]]
!CHECK: [[SPTRTYPE]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: [[SUBTYPE:![0-9]+]]
!CHECK: [[SUBTYPE]] = !DISubroutineType(types: [[ARGLIST:![0-9]+]]
!CHECK: [[ARGLIST]] = !{null, [[ARG1:![0-9]+]]
!CHECK: [[ARG1]] = !DICompositeType(tag: DW_TAG_array_type
!CHECK-SAME: dataLocation:
!CHECK-SAME: associated:

program main
  interface
    subroutine sub1(arg11)
      integer, pointer :: arg11(:)
    end subroutine
  end interface
  type type1
    procedure(sub1), pointer, nopass :: proc1
  end type
  type type2
    type(type1) :: mem1
  end type type2
  type(type2) :: arg1
end program main
