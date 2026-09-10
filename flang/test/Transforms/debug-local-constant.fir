// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

// Test that a named constant (Fortran PARAMETER) declared inside a procedure is
// described in the scope of that procedure. Two procedures declaring the same
// name must produce entries in distinct scopes, otherwise a debugger cannot
// tell them apart. Such a constant has internal linkage, so it is local to the
// compile unit and carries no linkage name.

module {
  func.func @_QPone() {
    %0 = fir.address_of(@_QFoneECq) : !fir.ref<i32>
    %1 = fircg.ext_declare %0 {uniq_name = "_QFoneECq"} : (!fir.ref<i32>) -> !fir.ref<i32> loc(#loc2)
    return
  } loc(#loc1)
  func.func @_QPtwo() {
    %0 = fir.address_of(@_QFtwoECq) : !fir.ref<i32>
    %1 = fircg.ext_declare %0 {uniq_name = "_QFtwoECq"} : (!fir.ref<i32>) -> !fir.ref<i32> loc(#loc4)
    return
  } loc(#loc3)
  fir.global internal @_QFoneECq constant : i32 {
    %c111_i32 = arith.constant 111 : i32
    fir.has_value %c111_i32 : i32
  } loc(#loc2)
  fir.global internal @_QFtwoECq constant : i32 {
    %c777_i32 = arith.constant 777 : i32
    fir.has_value %c777_i32 : i32
  } loc(#loc4)
}
#loc1 = loc("test.f90":1:1)
#loc2 = loc("test.f90":3:3)
#loc3 = loc("test.f90":7:1)
#loc4 = loc("test.f90":9:3)

// CHECK-DAG: #[[I4:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "integer(kind=4)", sizeInBits = 32, encoding = DW_ATE_signed>
// CHECK-DAG: #[[ONE:.*]] = #llvm.di_subprogram<{{.*}}name = "one"{{.*}}>
// CHECK-DAG: #[[TWO:.*]] = #llvm.di_subprogram<{{.*}}name = "two"{{.*}}>

// Each constant is scoped to its own procedure and, having internal linkage,
// is local to the unit and has no linkage name.
// CHECK-DAG: #llvm.di_global_variable<scope = #[[ONE]], name = "q", file = {{.*}}, line = 3, type = #[[I4]], isLocalToUnit = true, isDefined = true>
// CHECK-DAG: #llvm.di_global_variable<scope = #[[TWO]], name = "q", file = {{.*}}, line = 9, type = #[[I4]], isLocalToUnit = true, isDefined = true>
