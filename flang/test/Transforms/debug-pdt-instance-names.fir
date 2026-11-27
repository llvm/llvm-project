// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

// An instance of a parameterized type is named after its kind values, with a
// negative value spelled as in the uniqued name. A type with no kind values
// keeps its name.

module {
  func.func @_QPuser() {
    %0 = fir.alloca !fir.type<_QMmTt{i:i32}> {bindc_name = "x", uniq_name = "_QFuserEx"}
    %1 = fircg.ext_declare %0 uniq_name("_QFuserEx") : (!fir.ref<!fir.type<_QMmTt{i:i32}>>) -> !fir.ref<!fir.type<_QMmTt{i:i32}>> loc(#loc_x)
    %2 = fir.alloca !fir.type<_QMmTpdtK4{a:i32}> {bindc_name = "p4", uniq_name = "_QFuserEp4"}
    %3 = fircg.ext_declare %2 uniq_name("_QFuserEp4") : (!fir.ref<!fir.type<_QMmTpdtK4{a:i32}>>) -> !fir.ref<!fir.type<_QMmTpdtK4{a:i32}>> loc(#loc_p4)
    %4 = fir.alloca !fir.type<_QMmTpdtK8{a:i64}> {bindc_name = "p8", uniq_name = "_QFuserEp8"}
    %5 = fircg.ext_declare %4 uniq_name("_QFuserEp8") : (!fir.ref<!fir.type<_QMmTpdtK8{a:i64}>>) -> !fir.ref<!fir.type<_QMmTpdtK8{a:i64}>> loc(#loc_p8)
    %6 = fir.alloca !fir.type<_QMmTpdt2K8KN2{x:f64,n:i16}> {bindc_name = "q", uniq_name = "_QFuserEq"}
    %7 = fircg.ext_declare %6 uniq_name("_QFuserEq") : (!fir.ref<!fir.type<_QMmTpdt2K8KN2{x:f64,n:i16}>>) -> !fir.ref<!fir.type<_QMmTpdt2K8KN2{x:f64,n:i16}>> loc(#loc_q)
    return
  } loc(#loc_user)
}
#loc_user = loc("user.f90":1:1)
#loc_x = loc("user.f90":3:3)
#loc_p4 = loc("user.f90":4:3)
#loc_p8 = loc("user.f90":5:3)
#loc_q = loc("user.f90":6:3)

// CHECK-DAG: #llvm.di_composite_type<{{.*}}name = "t",
// CHECK-DAG: #llvm.di_composite_type<{{.*}}name = "PDTpdt_4",
// CHECK-DAG: #llvm.di_composite_type<{{.*}}name = "PDTpdt_8",
// CHECK-DAG: #llvm.di_composite_type<{{.*}}name = "PDTpdt2_8_N2",
