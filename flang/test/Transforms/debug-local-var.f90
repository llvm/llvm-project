! RUN: %flang_fc1 -emit-fir -debug-info-kind=standalone -mmlir --mlir-print-debuginfo %s -o - | \
! RUN: fir-opt --cg-rewrite="preserve-declare=true" --mlir-print-debuginfo | fir-opt --add-debug-info --mlir-print-debuginfo | FileCheck %s

! CHECK-DAG: #[[INT8:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "integer", sizeInBits = 64, encoding = DW_ATE_signed>
! CHECK-DAG: #[[INT4:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "integer", sizeInBits = 32, encoding = DW_ATE_signed>
! CHECK-DAG: #[[REAL8:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real", sizeInBits = 64, encoding = DW_ATE_float>
! CHECK-DAG: #[[LOG1:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "logical", sizeInBits = 8, encoding = DW_ATE_boolean>
! CHECK-DAG: #[[REAL4:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real", sizeInBits = 32, encoding = DW_ATE_float>
! CHECK-DAG: #[[LOG4:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "logical", sizeInBits = 32, encoding = DW_ATE_boolean>
! CHECK-DAG: #[[MAIN:.*]] = #llvm.di_subprogram<{{.*}}name = "_QQmain"{{.*}}>
! CHECK-DAG: #[[FN1:.*]] = #llvm.di_subprogram<{{.*}}name = "fn1"{{.*}}>
! CHECK-DAG: #[[FN2:.*]] = #llvm.di_subprogram<{{.*}}name = "fn2"{{.*}}>

program mn
! CHECK-DAG: #[[I4:.*]] = #llvm.di_local_variable<scope = #[[MAIN]], name = "i4", file = #{{.*}}, line = [[@LINE+6]], type = #[[INT4]]>
! CHECK-DAG: #[[I8:.*]] = #llvm.di_local_variable<scope = #[[MAIN]], name = "i8", file = #{{.*}}, line = [[@LINE+6]], type = #[[INT8]]>
! CHECK-DAG: #[[R4:.*]] = #llvm.di_local_variable<scope = #[[MAIN]], name = "r4", file = #{{.*}}, line = [[@LINE+6]], type = #[[REAL4]]>
! CHECK-DAG: #[[R8:.*]] = #llvm.di_local_variable<scope = #[[MAIN]], name = "r8", file = #{{.*}}, line = [[@LINE+6]], type = #[[REAL8]]>
! CHECK-DAG: #[[L1:.*]] = #llvm.di_local_variable<scope = #[[MAIN]], name = "l1", file = #{{.*}}, line = [[@LINE+6]], type = #[[LOG1]]>
! CHECK-DAG: #[[L4:.*]] = #llvm.di_local_variable<scope = #[[MAIN]], name = "l4", file = #{{.*}}, line = [[@LINE+6]], type = #[[LOG4]]>
  integer(kind=4) :: i4
  integer(kind=8) :: i8
  real(kind=4) :: r4
  real(kind=8) :: r8
  logical(kind=1) :: l1
  logical(kind=4) :: l4
  i8 = fn1(i4, r8, l1)
  i4 = fn2(i8, r4, l4)
contains
  function fn1(a1, b1, c1) result (res1)
! CHECK-DAG: #[[A1:.*]] = #llvm.di_local_variable<scope = #[[FN1]], name = "a1", file = #{{.*}}, line = [[@LINE+4]], arg = 1, type = #[[INT4]]>
! CHECK-DAG: #[[B1:.*]] = #llvm.di_local_variable<scope = #[[FN1]], name = "b1", file = #{{.*}}, line = [[@LINE+4]], arg = 2, type = #[[REAL8]]>
! CHECK-DAG: #[[C1:.*]] = #llvm.di_local_variable<scope = #[[FN1]], name = "c1", file = #{{.*}}, line = [[@LINE+4]], arg = 3, type = #[[LOG1]]>
! CHECK-DAG: #[[RES1:.*]] = #llvm.di_local_variable<scope = #[[FN1]], name = "res1", file = #{{.*}}, line = [[@LINE+4]], type = #[[INT8]]>
    integer(kind=4), intent(in) :: a1
    real(kind=8), intent(in) :: b1
    logical(kind=1), intent(in) :: c1
    integer(kind=8) :: res1
    res1 = a1 + b1
  end function

  function fn2(a2, b2, c2) result (res2)
    implicit none
! CHECK-DAG: #[[A2:.*]] = #llvm.di_local_variable<scope = #[[FN2]], name = "a2", file = #{{.*}}, line = [[@LINE+4]], arg = 1, type = #[[INT8]]>
! CHECK-DAG: #[[B2:.*]] = #llvm.di_local_variable<scope = #[[FN2]], name = "b2", file = #{{.*}}, line = [[@LINE+4]], arg = 2, type = #[[REAL4]]>
! CHECK-DAG: #[[C2:.*]] = #llvm.di_local_variable<scope = #[[FN2]], name = "c2", file = #{{.*}}, line = [[@LINE+4]], arg = 3, type = #[[LOG4]]>
! CHECK-DAG: #[[RES2:.*]] = #llvm.di_local_variable<scope = #[[FN2]], name = "res2", file = #{{.*}}, line = [[@LINE+4]], type = #[[INT4]]>
    integer(kind=8), intent(in) :: a2
    real(kind=4), intent(in) :: b2
    logical(kind=4), intent(in) :: c2
    integer(kind=4) :: res2
    res2 = a2 + b2
  end function
end program
