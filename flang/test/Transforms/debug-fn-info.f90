! RUN: %flang_fc1 -emit-fir -debug-info-kind=standalone -mmlir --mlir-print-debuginfo %s -o - | fir-opt --add-debug-info --mlir-print-debuginfo  | FileCheck %s


! CHECK-DAG: #[[INT8:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "integer", sizeInBits = 64, encoding = DW_ATE_signed>
! CHECK-DAG: #[[INT4:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "integer", sizeInBits = 32, encoding = DW_ATE_signed>
! CHECK-DAG: #[[REAL8:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real", sizeInBits = 64, encoding = DW_ATE_float>
! CHECK-DAG: #[[LOG1:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "logical", sizeInBits = 8, encoding = DW_ATE_boolean>
! CHECK-DAG: #[[REAL4:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real", sizeInBits = 32, encoding = DW_ATE_float>
! CHECK-DAG: #[[LOG4:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "logical", sizeInBits = 32, encoding = DW_ATE_boolean>
! CHECK: #[[TY0:.*]] = #llvm.di_subroutine_type<callingConvention = DW_CC_program>
! CHECK: #[[TY1:.*]] = #llvm.di_subroutine_type<callingConvention = DW_CC_normal, types = #[[INT8]], #[[INT4]], #[[REAL8]], #[[LOG1]]>
! CHECK: #[[TY2:.*]] = #llvm.di_subroutine_type<callingConvention = DW_CC_normal, types = #[[INT4]], #[[INT8]], #[[REAL4]], #[[LOG4]]>

! CHECK: #di_subprogram1 = #llvm.di_subprogram<id = {{.*}}, compileUnit = {{.*}}, scope = {{.*}}, name = "_QQmain", linkageName = "_QQmain", file = {{.*}}, line = [[@LINE+1]], scopeLine = [[@LINE+1]], subprogramFlags = Definition, type = #[[TY0]]>
program mn
  integer(kind=4) :: i4
  integer(kind=8) :: i8
  real(kind=4) :: r4
  real(kind=8) :: r8
  logical(kind=1) :: l1
  logical(kind=4) :: l4
  i8 = fn1(i4, r8, l1)
  i4 = fn2(i8, r4, l4)
contains
  ! CHECK: #di_subprogram2 = #llvm.di_subprogram<id = {{.*}}, compileUnit = {{.*}}, scope = {{.*}}, name = "fn1", linkageName = "_QFPfn1", file = {{.*}}, line = [[@LINE+1]], scopeLine = [[@LINE+1]], subprogramFlags = Definition, type = #[[TY1]]>
  function fn1(a, b, c) result (res)
    implicit none
    integer(kind=4), intent(in) :: a
    real(kind=8), intent(in) :: b
    logical(kind=1), intent(in) :: c
    integer(kind=8) :: res
    res = a + b
  end function

! CHECK: #di_subprogram3 = #llvm.di_subprogram<id = {{.*}}, compileUnit = {{.*}}, scope = {{.*}}, name = "fn2", linkageName = "_QFPfn2", file = {{.*}}, line = [[@LINE+1]], scopeLine = [[@LINE+1]], subprogramFlags = Definition, type = #[[TY2]]>
  function fn2(a, b, c) result (res)
    implicit none
    integer(kind=8), intent(in) :: a
    real(kind=4), intent(in) :: b
    logical(kind=4), intent(in) :: c
    integer(kind=4) :: res
    res = a + b
  end function
end program

