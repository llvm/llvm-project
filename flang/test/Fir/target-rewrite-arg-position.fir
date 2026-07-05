// RUN: fir-opt --target-rewrite="target=x86_64-unknown-linux-gnu" %s | FileCheck %s

// Test with an argument shift.

func.func @_QFPf(%arg0: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc}) -> complex<f128> {
  %0 = fir.alloca complex<f128> {bindc_name = "f", uniq_name = "_QFfEf"}
  %c2_i32 = arith.constant 2 : i32
  %1 = fir.convert %c2_i32 : (i32) -> f128
  %cst = arith.constant 0.000000e+00 : f128
  %2 = fir.undefined complex<f128>
  %3 = fir.insert_value %2, %1, [0 : index] : (complex<f128>, f128) -> complex<f128>
  %4 = fir.insert_value %3, %cst, [1 : index] : (complex<f128>, f128) -> complex<f128>
  fir.store %4 to %0 : !fir.ref<complex<f128>>
  %5 = fir.load %0 : !fir.ref<complex<f128>>
  return %5 : complex<f128>
}

// CHECK-LABEL: func.func @_QFPf
// CHECK-SAME:    %{{.*}}: !fir.ref<tuple<f128, f128>> {llvm.align = 16 : i32, llvm.sret = tuple<f128, f128>}, %arg1: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc, llvm.nest}) {

// -----

// Test with no shift.

func.func @_QFPs(%arg0: !fir.ref<i32> {fir.host_assoc}) {
  return
}

// CHECK: func.func @_QFPs(%arg0: !fir.ref<i32> {fir.host_assoc, llvm.nest})
