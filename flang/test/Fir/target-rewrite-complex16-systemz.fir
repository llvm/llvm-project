// RUN: fir-opt --target-rewrite="target=s390x-unknown-linux-gnu" %s | FileCheck %s
// This test checks SystemZ target rewriting for complex(16) / complex<f128>.
// An explicit `llvm.data_layout` is required to override the generic
// MLIR fallback data layout (which defaults f128 to 16-byte alignment).
// The SystemZ ABI requires an 8-byte alignment for complex(16) aggregates.
module attributes {
  llvm.data_layout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64",
  llvm.target_triple = "s390x-unknown-linux-gnu" } {

func.func @returncomplex16() -> complex<f128> {
  %1 = fir.undefined complex<f128>
  %2 = arith.constant 2.0 : f128
  %3 = fir.convert %2 : (f128) -> f128
  %c0 = arith.constant 0 : i32
  %4 = fir.insert_value %1, %3, [0 : i32] : (complex<f128>, f128) -> complex<f128>
  %c1 = arith.constant 1 : i32
  %5 = arith.constant -42.0 : f128
  %6 = fir.insert_value %4, %5, [1 : i32] : (complex<f128>, f128) -> complex<f128>
  return %6 : complex<f128>
}

func.func private @paramcomplex16(complex<f128>) -> ()

func.func @callcomplex16() {
  %1 = fir.call @returncomplex16() : () -> complex<f128>
  fir.call @paramcomplex16(%1) : (complex<f128>) -> ()
  return
}

func.func private @calleemultipleparamscomplex16(complex<f128>, complex<f128>, complex<f128>) -> ()

func.func @multipleparamscomplex16(%z1 : complex<f128>, %z2 : complex<f128>, %z3 : complex<f128>) {
  fir.call @calleemultipleparamscomplex16(%z1, %z2, %z3) : (complex<f128>, complex<f128>, complex<f128>) -> ()
  return
}

func.func private @mlircomplexf128(%z1: complex<f128>, %z2: complex<f128>) -> complex<f128> {
  %0 = fir.call @mlircomplexf128(%z1, %z2) : (complex<f128>, complex<f128>) -> complex<f128>
  return %0 : complex<f128>
}

func.func @addrof() {
  %r = fir.address_of(@returncomplex16) : () -> complex<f128>
  %p = fir.address_of(@paramcomplex16) : (complex<f128>) -> ()
  return
}

} // module

// CHECK-LABEL: func.func @returncomplex16(
// CHECK-SAME:    %{{.*}}: !fir.ref<tuple<f128, f128>> {llvm.align = 8 : i32, llvm.sret = tuple<f128, f128>}) {

// CHECK: func.func private @paramcomplex16(!fir.ref<tuple<f128, f128>> {llvm.align = 8 : i32, llvm.byval = tuple<f128, f128>})

// CHECK-LABEL: func.func @callcomplex16() {
// CHECK:         fir.call @returncomplex16(%{{.*}}) : (!fir.ref<tuple<f128, f128>> {llvm.align = 8 : i32, llvm.sret = tuple<f128, f128>}) -> ()
// CHECK:         fir.call @paramcomplex16(%{{.*}}) : (!fir.ref<tuple<f128, f128>> {llvm.align = 8 : i32, llvm.byval = tuple<f128, f128>}) -> ()

// CHECK: func.func private @calleemultipleparamscomplex16(!fir.ref<tuple<f128, f128>> {llvm.align = 8 : i32, llvm.byval = tuple<f128, f128>}, !fir.ref<tuple<f128, f128>> {llvm.align = 8 : i32, llvm.byval = tuple<f128, f128>}, !fir.ref<tuple<f128, f128>> {llvm.align = 8 : i32, llvm.byval = tuple<f128, f128>})

// CHECK-LABEL: func.func @multipleparamscomplex16(
// CHECK-SAME:    %{{.*}}: !fir.ref<tuple<f128, f128>> {llvm.align = 8 : i32, llvm.byval = tuple<f128, f128>}, %{{.*}}: !fir.ref<tuple<f128, f128>> {llvm.align = 8 : i32, llvm.byval = tuple<f128, f128>}, %{{.*}}: !fir.ref<tuple<f128, f128>> {llvm.align = 8 : i32, llvm.byval = tuple<f128, f128>}) {

// CHECK-LABEL: func.func private @mlircomplexf128(
// CHECK-SAME:    %{{.*}}: !fir.ref<tuple<f128, f128>> {llvm.align = 8 : i32, llvm.sret = tuple<f128, f128>}, %{{.*}}: !fir.ref<tuple<f128, f128>> {llvm.align = 8 : i32, llvm.byval = tuple<f128, f128>}, %{{.*}}: !fir.ref<tuple<f128, f128>> {llvm.align = 8 : i32, llvm.byval = tuple<f128, f128>}) {

// CHECK-LABEL: func.func @addrof() {
// CHECK:         fir.address_of(@returncomplex16) : (!fir.ref<tuple<f128, f128>>) -> ()
// CHECK:         fir.address_of(@paramcomplex16) : (!fir.ref<tuple<f128, f128>>) -> ()
