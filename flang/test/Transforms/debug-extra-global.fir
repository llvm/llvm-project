// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  fir.global linkonce_odr @_QFEXnXxcx constant target : !fir.char<1,3> {
    %0 = fir.string_lit "xcx"(3) : !fir.char<1,3>
    fir.has_value %0 : !fir.char<1,3>
  } loc(#loc1)
  fir.global linkonce_odr @_QFEXnXxdtx constant target : !fir.char<1,4> {
    %0 = fir.string_lit "xdtx"(4) : !fir.char<1,4>
    fir.has_value %0 : !fir.char<1,4>
  } loc(#loc1)
}
#loc1 = loc("derived.f90":24:1)

// Test that no di_global_variable gets created for these compile generated
// globals.

// CHECK-NOT: #di_global_variable
