// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s -o - | FileCheck %s

// This test checks that debug information for fir.real type works ok.

module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  func.func @_QPfn1(%arg0: !fir.ref<!fir.complex<8>> {fir.bindc_name = "a"} ) {
    %0 = fir.declare %arg0 {uniq_name = "_QFfn1Ea"} : (!fir.ref<!fir.complex<8>>) -> !fir.ref<!fir.complex<8>>
    %1 = fir.alloca f32 {bindc_name = "abserror", uniq_name = "_QFfn1Eabserror"}
    %2 = fir.declare %1 {uniq_name = "_QFfn1Eabserror"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %3 = fir.load %0 : !fir.ref<!fir.complex<8>>
    %4 = fir.extract_value %3, [0 : i32] : (!fir.complex<8>) -> !fir.real<8>
    %5 = fir.extract_value %3, [1 : i32] : (!fir.complex<8>) -> !fir.real<8>
    %6 = fir.call @cabs(%4, %5) : (!fir.real<8>, !fir.real<8>) -> f64
    %7 = fir.convert %6 : (f64) -> f32
    fir.store %7 to %2 : !fir.ref<f32>
    return
  } loc(#loc1)
  func.func private @cabs(!fir.real<8>, !fir.real<8>) -> f64 attributes {fir.bindc_name = "cabs", fir.runtime}
} loc(#loc)
#loc1 = loc("test.f90":5:1)
#loc = loc("test.f90":0:0)

// CHECK-DAG: #[[TY:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real", sizeInBits = 64, encoding = DW_ATE_float>
// CHECK-DAG: #[[TY1:.*]] = #llvm.di_subroutine_type<callingConvention = DW_CC_normal, types = #[[TY]], #[[TY]], #[[TY]]>
// CHECK-DAG: #{{.*}} = #llvm.di_subprogram<scope = #{{.*}}, name = "cabs", linkageName = "cabs", file = #{{.*}}, line = {{.*}}, scopeLine = {{.*}}, type = #[[TY1]]>
