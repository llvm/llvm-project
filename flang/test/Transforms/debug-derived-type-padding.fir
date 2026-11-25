// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

// The size of a derived type or tuple includes the padding after its last
// component that keeps the elements of an array of it aligned. A packed type
// has no such padding, and does not align its components, though each still
// takes up its allocation size.

module {
  func.func @_QPuser() {
    %0 = fir.alloca !fir.type<_QMmTt{x:f64,n:i16}> {bindc_name = "x", uniq_name = "_QFuserEx"}
    %1 = fircg.ext_declare %0 uniq_name("_QFuserEx") : (!fir.ref<!fir.type<_QMmTt{x:f64,n:i16}>>) -> !fir.ref<!fir.type<_QMmTt{x:f64,n:i16}>> loc(#loc_x)
    %2 = fir.alloca !fir.type<_QMmTp<{x:f64,n:i16}>> {bindc_name = "p", uniq_name = "_QFuserEp"}
    %3 = fircg.ext_declare %2 uniq_name("_QFuserEp") : (!fir.ref<!fir.type<_QMmTp<{x:f64,n:i16}>>>) -> !fir.ref<!fir.type<_QMmTp<{x:f64,n:i16}>>> loc(#loc_p)
    %4 = fir.alloca !fir.type<_QMmTq<{n:i16,x:f64}>> {bindc_name = "q", uniq_name = "_QFuserEq"}
    %5 = fircg.ext_declare %4 uniq_name("_QFuserEq") : (!fir.ref<!fir.type<_QMmTq<{n:i16,x:f64}>>>) -> !fir.ref<!fir.type<_QMmTq<{n:i16,x:f64}>>> loc(#loc_q)
    %6 = fir.alloca !fir.type<_QMmTr<{a:f80,n:i16}>> {bindc_name = "r", uniq_name = "_QFuserEr"}
    %7 = fircg.ext_declare %6 uniq_name("_QFuserEr") : (!fir.ref<!fir.type<_QMmTr<{a:f80,n:i16}>>>) -> !fir.ref<!fir.type<_QMmTr<{a:f80,n:i16}>>> loc(#loc_r)
    return
  } loc(#loc_user)
  func.func private @fn(!fir.ref<tuple<f64, i16>>)
}
#loc_user = loc("user.f90":1:1)
#loc_x = loc("user.f90":3:3)
#loc_p = loc("user.f90":4:3)
#loc_q = loc("user.f90":5:3)
#loc_r = loc("user.f90":6:3)

// CHECK-DAG: #llvm.di_composite_type<{{.*}}name = "t", {{.*}}sizeInBits = 128,
// CHECK-DAG: #llvm.di_composite_type<{{.*}}name = "p", {{.*}}sizeInBits = 80,
// CHECK-DAG: #llvm.di_composite_type<tag = DW_TAG_structure_type, name = "", {{.*}}sizeInBits = 128,

// CHECK-DAG: #[[QN:.+]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "n", baseType = #{{.+}}, sizeInBits = 16, alignInBits = 8>
// CHECK-DAG: #[[QX:.+]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "x", baseType = #{{.+}}, sizeInBits = 64, alignInBits = 8, offsetInBits = 16>
// CHECK-DAG: #llvm.di_composite_type<{{.*}}name = "q", {{.*}}sizeInBits = 80, elements = #[[QN]], #[[QX]]>

// CHECK-DAG: #[[RN:.+]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "n", baseType = #{{.+}}, sizeInBits = 16, alignInBits = 8, offsetInBits = 128>
// CHECK-DAG: #llvm.di_composite_type<{{.*}}name = "r", {{.*}}sizeInBits = 144, elements = #{{.+}}, #[[RN]]>
