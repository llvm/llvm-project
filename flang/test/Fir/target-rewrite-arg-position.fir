// RUN: fir-opt --target-rewrite="target=x86_64-unknown-linux-gnu" %s | FileCheck %s

// Test with an argument shift.

func.func @_QFPf(%arg0: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc}) -> !fir.complex<16> {
  %0 = fir.alloca !fir.complex<16> {bindc_name = "f", uniq_name = "_QFfEf"}
  %c2_i32 = arith.constant 2 : i32
  %1 = fir.convert %c2_i32 : (i32) -> f128
  %cst = arith.constant 0.000000e+00 : f128
  %2 = fir.undefined !fir.complex<16>
  %3 = fir.insert_value %2, %1, [0 : index] : (!fir.complex<16>, f128) -> !fir.complex<16>
  %4 = fir.insert_value %3, %cst, [1 : index] : (!fir.complex<16>, f128) -> !fir.complex<16>
  fir.store %4 to %0 : !fir.ref<!fir.complex<16>>
  %5 = fir.load %0 : !fir.ref<!fir.complex<16>>
  return %5 : !fir.complex<16>
}

// CHECK-LABEL: func.func @_QFPf
// CHECK-SAME:    %{{.*}}: !fir.ref<tuple<!fir.real<16>, !fir.real<16>>> {llvm.align = 16 : i32, llvm.sret = tuple<!fir.real<16>, !fir.real<16>>}, %arg1: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc, llvm.nest}) {

// -----

// Test with no shift.

func.func @_QFPs(%arg0: !fir.ref<i32> {fir.host_assoc}) {
  return
}

// CHECK: func.func @_QFPs(%arg0: !fir.ref<i32> {fir.host_assoc, llvm.nest})
