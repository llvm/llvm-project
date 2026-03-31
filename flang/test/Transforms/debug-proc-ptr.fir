// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

module {
  func.func @_QQmain() attributes {fir.bindc_name = "test"} {
    %0 = fir.alloca (!fir.ref<i32>) -> i32 {bindc_name = "fun_ptr", uniq_name = "_QFEfun_ptr"}
    %1 = fircg.ext_declare %0 {uniq_name = "_QFEfun_ptr"} : (!fir.ref<(!fir.ref<i32>) -> i32>) -> !fir.ref<(!fir.ref<i32>) -> i32> loc(#loc1)

    // Procedure pointer with no return: procedure(sub1), pointer :: sub_ptr
    %2 = fir.alloca () -> () {bindc_name = "sub_ptr", uniq_name = "_QFEsub_ptr"}
    %3 = fircg.ext_declare %2 {uniq_name = "_QFEsub_ptr"} : (!fir.ref<() -> ()>) -> !fir.ref<() -> ()> loc(#loc2)

    // Procedure pointer with multiple args: procedure(func2), pointer :: func_ptr
    %4 = fir.alloca (!fir.ref<i32>, !fir.ref<f64>) -> f32 {bindc_name = "func_ptr", uniq_name = "_QFEfunc_ptr"}
    %5 = fircg.ext_declare %4 {uniq_name = "_QFEfunc_ptr"} : (!fir.ref<(!fir.ref<i32>, !fir.ref<f64>) -> f32>) -> !fir.ref<(!fir.ref<i32>, !fir.ref<f64>) -> f32> loc(#loc3)

    return
  } loc(#loc)
}
#loc = loc("test.f90":1:1)
#loc1 = loc("test.f90":2:30)
#loc2 = loc("test.f90":3:30)
#loc3 = loc("test.f90":4:30)

// CHECK-DAG: #[[INT:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "integer", sizeInBits = 32, encoding = DW_ATE_signed>
// CHECK-DAG: #[[REAL32:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real", sizeInBits = 32, encoding = DW_ATE_float>
// CHECK-DAG: #[[REAL:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real(kind=8)", sizeInBits = 64, encoding = DW_ATE_float>

// CHECK-DAG: #[[PTR_INT:.*]] = #llvm.di_derived_type<tag = DW_TAG_pointer_type{{.*}}baseType = #[[INT]]{{.*}}>
// CHECK-DAG: #[[PTR_REAL:.*]] = #llvm.di_derived_type<tag = DW_TAG_pointer_type{{.*}}baseType = #[[REAL]]{{.*}}>

// CHECK-DAG: #[[SUB1:.*]] = #llvm.di_subroutine_type<types = #[[INT]], #[[PTR_INT]]>
// CHECK-DAG: #[[PTR_SUB1:.*]] = #llvm.di_derived_type<tag = DW_TAG_pointer_type{{.*}}baseType = #[[SUB1]]{{.*}}>
// CHECK-DAG: #llvm.di_local_variable<{{.*}}name = "fun_ptr"{{.*}}type = #[[PTR_SUB1]]{{.*}}>

// CHECK-DAG: #di_subroutine_type{{.*}} = #llvm.di_subroutine_type<types = #di_null_type>
// CHECK-DAG: #di_local_variable{{.*}} = #llvm.di_local_variable<{{.*}}name = "sub_ptr"{{.*}}type = #di_derived_type{{.*}}>
// CHECK-DAG: #di_derived_type{{.*}} = #llvm.di_derived_type<tag = DW_TAG_pointer_type{{.*}}baseType = #di_subroutine_type{{.*}}{{.*}}>

// CHECK-DAG: #[[SUB3:.*]] = #llvm.di_subroutine_type<types = #[[REAL32]], #[[PTR_INT]], #[[PTR_REAL]]>
// CHECK-DAG: #[[PTR_SUB3:.*]] = #llvm.di_derived_type<tag = DW_TAG_pointer_type{{.*}}baseType = #[[SUB3]]{{.*}}>
// CHECK-DAG: #llvm.di_local_variable<{{.*}}name = "func_ptr"{{.*}}type = #[[PTR_SUB3]]{{.*}}>
