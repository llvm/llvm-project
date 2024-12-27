// RUN: tco --target=aarch64-unknown-linux-gnu %s | FileCheck %s --check-prefix=AARCH64

// AARCH64-LABEL: define { fp128, fp128 } @gen16()
func.func @gen16() -> complex<f128> {
  // AARCH64: %[[VAL1:.*]] = alloca { fp128, fp128 }, i64 1, align 16
  %1 = fir.undefined complex<f128>
  %2 = arith.constant 1.0 : f128
  %3 = arith.constant -4.0 : f128
  %c0 = arith.constant 0 : i32
  // AARCH64: store { fp128, fp128 } { fp128 0xL0000000000000000C001000000000000, fp128 0xL00000000000000003FFF000000000000 }, ptr %[[VAL1]], align 16
  %4 = fir.insert_value %1, %3, [0 : index] : (complex<f128>, f128) -> complex<f128>
  %c1 = arith.constant 1 : i32
  %5 = fir.insert_value %4, %2, [1 : index] : (complex<f128>, f128) -> complex<f128>
  // AARCH64: %[[VAL2:.*]] = load { fp128, fp128 }, ptr %[[VAL1]], align 16
  // AARCH64: ret { fp128, fp128 } %[[VAL2]]
  return %5 : complex<f128>
}

// AARCH64: declare void @sink16([2 x fp128])
func.func private @sink16(complex<f128>) -> ()

// AARCH64-LABEL: define void @call16()
func.func @call16() {
  // AARCH64: = call { fp128, fp128 } @gen16()
  %1 = fir.call @gen16() : () -> complex<f128>
  // AARCH64: call void @sink16([2 x fp128] %
  fir.call @sink16(%1) : (complex<f128>) -> ()
  return
}
