! RUN: %flang_fc1 -emit-fir -debug-info-kind=standalone -mmlir --mlir-print-debuginfo %s -o - | \
! RUN: fir-opt --cg-rewrite="preserve-declare=true" --mlir-print-debuginfo | fir-opt --add-debug-info --mlir-print-debuginfo | FileCheck %s


program mn
  integer d1(3)
  integer d2(2, 5)
  real d3(6, 8, 7)

  i8 = fn1(d1, d2, d3)
contains
  function fn1(a1, b1, c1) result (res)
    integer a1(3)
    integer b1(2, 5)

    real c1(6, 8, 7)
    integer res
    res = a1(1) + b1(1,2) + c1(3, 3, 4)
  end function

end program

! CHECK-DAG: #[[INT:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "integer", sizeInBits = 32, encoding = DW_ATE_signed>
! CHECK-DAG: #[[REAL:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real", sizeInBits = 32, encoding = DW_ATE_float>
! CHECK-DAG: #[[D1TY:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type{{.*}}baseType = #[[INT]], elements = #llvm.di_subrange<count = 3 : i64, lowerBound = 1 : i64>>
! CHECK-DAG: #[[D2TY:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type{{.*}}baseType = #[[INT]], elements = #llvm.di_subrange<count = 2 : i64, lowerBound = 1 : i64>, #llvm.di_subrange<count = 5 : i64, lowerBound = 1 : i64>>
! CHECK-DAG: #[[D3TY:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type{{.*}}baseType = #[[REAL]], elements = #llvm.di_subrange<count = 6 : i64, lowerBound = 1 : i64>, #llvm.di_subrange<count = 8 : i64, lowerBound = 1 : i64>, #llvm.di_subrange<count = 7 : i64, lowerBound = 1 : i64>>
! CHECK-DAG: #llvm.di_local_variable<{{.*}}name = "d1"{{.*}}type = #[[D1TY]]>
! CHECK-DAG: #llvm.di_local_variable<{{.*}}name = "d2"{{.*}}type = #[[D2TY]]>
! CHECK-DAG: #llvm.di_local_variable<{{.*}}name = "d3"{{.*}}type = #[[D3TY]]>
! CHECK-DAG: #llvm.di_local_variable<{{.*}}name = "a1"{{.*}}type = #[[D1TY]]>
! CHECK-DAG: #llvm.di_local_variable<{{.*}}name = "b1"{{.*}}type = #[[D2TY]]>
! CHECK-DAG: #llvm.di_local_variable<{{.*}}name = "c1"{{.*}}type = #[[D3TY]]>
