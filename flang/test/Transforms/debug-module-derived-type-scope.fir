// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

// Test that a derived type defined in a module gets the scope of that module.

module {
  fir.module_debug_imports "defined_m" {
  } loc(#loc_defined_m)
  fir.global @_QMdefined_mEcounter : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  } loc(#loc_counter)
  fir.module_debug_imports "impl" in "defined_m" parent "defined_m" {
  } loc(#loc_impl)

  func.func @_QPuser() {
    %0 = fir.alloca !fir.type<_QMdefined_mTt{i:i32}> {bindc_name = "x", uniq_name = "_QFuserEx"}
    %1 = fircg.ext_declare %0 uniq_name("_QFuserEx") : (!fir.ref<!fir.type<_QMdefined_mTt{i:i32}>>) -> !fir.ref<!fir.type<_QMdefined_mTt{i:i32}>> loc(#loc_x)
    %2 = fir.alloca !fir.type<_QMused_mTnode{l:!fir.type<_QMother_mTleaf{v:i32}>}> {bindc_name = "n", uniq_name = "_QFuserEn"}
    %3 = fircg.ext_declare %2 uniq_name("_QFuserEn") : (!fir.ref<!fir.type<_QMused_mTnode{l:!fir.type<_QMother_mTleaf{v:i32}>}>>) -> !fir.ref<!fir.type<_QMused_mTnode{l:!fir.type<_QMother_mTleaf{v:i32}>}>> loc(#loc_n)
    %4 = fir.alloca !fir.type<_QFuserTuser_t{w:i32}> {bindc_name = "u", uniq_name = "_QFuserEu"}
    %5 = fircg.ext_declare %4 uniq_name("_QFuserEu") : (!fir.ref<!fir.type<_QFuserTuser_t{w:i32}>>) -> !fir.ref<!fir.type<_QFuserTuser_t{w:i32}>> loc(#loc_u)
    return
  } loc(#loc_user)

  func.func @_QMdefined_mPmsub() {
    %0 = fir.alloca !fir.type<_QMdefined_mFmsubTmsub_t{w:i32}> {bindc_name = "m", uniq_name = "_QMdefined_mFmsubEm"}
    %1 = fircg.ext_declare %0 uniq_name("_QMdefined_mFmsubEm") : (!fir.ref<!fir.type<_QMdefined_mFmsubTmsub_t{w:i32}>>) -> !fir.ref<!fir.type<_QMdefined_mFmsubTmsub_t{w:i32}>> loc(#loc_m)
    return
  } loc(#loc_msub)

  func.func @_QMdefined_mPsmod_sub() attributes {fir.defining_submodule = "impl"} {
    %0 = fir.alloca !fir.type<_QMdefined_mSimplTsub_t{s:i32}> {bindc_name = "s", uniq_name = "_QMdefined_mSimplFsmod_subEs"}
    %1 = fircg.ext_declare %0 uniq_name("_QMdefined_mSimplFsmod_subEs") : (!fir.ref<!fir.type<_QMdefined_mSimplTsub_t{s:i32}>>) -> !fir.ref<!fir.type<_QMdefined_mSimplTsub_t{s:i32}>> loc(#loc_s)
    return
  } loc(#loc_smod_sub)
}
#loc_defined_m = loc("defined.f90":1:1)
#loc_counter = loc("defined.f90":5:14)
#loc_msub = loc("defined.f90":10:3)
#loc_m = loc("defined.f90":14:5)
#loc_impl = loc("defined.f90":20:1)
#loc_smod_sub = loc("defined.f90":25:3)
#loc_s = loc("defined.f90":26:5)
#loc_user = loc("user.f90":1:1)
#loc_x = loc("user.f90":3:3)
#loc_n = loc("user.f90":4:3)
#loc_u = loc("user.f90":8:3)

// CHECK-DAG: #[[CU:.+]] = #llvm.di_compile_unit<recId = {{.+}}, id = {{.+}}>
// CHECK-DAG: #[[DEFINED:.+]] = #llvm.di_module<{{.*}}scope = #[[CU]], name = "defined_m", line = 1>
// CHECK-DAG: #[[IMPL:.+]] = #llvm.di_module<{{.*}}scope = #[[CU]], name = "defined_m.impl", line = 20>
// CHECK-DAG: #[[USED:.+]] = #llvm.di_module<name = "used_m", isDecl = true>
// CHECK-DAG: #[[OTHER:.+]] = #llvm.di_module<name = "other_m", isDecl = true>
// CHECK-DAG: #[[USER:.+]] = #llvm.di_subprogram<{{.*}}name = "user"
// CHECK-DAG: #[[MSUB:.+]] = #llvm.di_subprogram<{{.*}}scope = #[[DEFINED]], name = "msub"

// The scope of a type is the module that the module's variables are scoped in.
// CHECK-DAG: #llvm.di_global_variable<scope = #[[DEFINED]], name = "counter"
// CHECK-DAG: #llvm.di_composite_type<{{.*}}name = "t", {{.*}}scope = #[[DEFINED]],

// A type from a module only used here, and one used only as a component type.
// CHECK-DAG: #llvm.di_composite_type<{{.*}}name = "node", {{.*}}scope = #[[USED]],
// CHECK-DAG: #llvm.di_composite_type<{{.*}}name = "leaf", {{.*}}scope = #[[OTHER]],

// A type defined in a submodule gets the submodule.
// CHECK-DAG: #llvm.di_composite_type<{{.*}}name = "sub_t", {{.*}}scope = #[[IMPL]],

// A type defined in a procedure, even a module procedure, stays in it.
// CHECK-DAG: #llvm.di_composite_type<{{.*}}name = "user_t", {{.*}}scope = #[[USER]],
// CHECK-DAG: #llvm.di_composite_type<{{.*}}name = "msub_t", {{.*}}scope = #[[MSUB]],
