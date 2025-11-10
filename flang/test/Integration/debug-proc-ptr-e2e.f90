! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck %s

program test_proc_ptr
  implicit none
  procedure(fun1), pointer :: fun_ptr

  fun_ptr => fun1
  print *, fun_ptr(3)

contains
  integer function fun1(x)
    integer :: x
    fun1 = x + 1
  end function fun1
end program test_proc_ptr

! Check that fun_ptr is declared with correct type
! CHECK-DAG: ![[INT:.*]] = !DIBasicType(name: "integer", size: 32, encoding: DW_ATE_signed)
! CHECK-DAG: ![[PTR_INT:.*]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[INT]], size: 64)

! Check that fun_ptr variable is a pointer to a subroutine type
! The order is: DILocalVariable -> pointer type -> subroutine type -> {return, params}
! CHECK-DAG: ![[FUN_PTR_VAR:.*]] = !DILocalVariable(name: "fun_ptr", {{.*}}type: ![[PROC_PTR:[0-9]+]]
! CHECK-DAG: ![[PROC_PTR]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[SUBR_TYPE:[0-9]+]], size: 64)
! CHECK-DAG: ![[SUBR_TYPE]] = !DISubroutineType(types: ![[SUBR_TYPES:[0-9]+]])
! CHECK-DAG: ![[SUBR_TYPES]] = !{![[INT]], ![[PTR_INT]]}
