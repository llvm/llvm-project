/// Test i32 passing and returning on LoongArch64
/// LoongArch64 LP64D ABI requires unsigned 32 bit integers to be sign extended.

// REQUIRES: loongarch-registered-target
// RUN: fir-opt --target-rewrite="target=loongarch64-unknown-linux-gnu" %s | FileCheck %s --check-prefix=LOONGARCH64
// RUN: tco -target="loongarch64-unknown-linux-gnu" %s | FileCheck %s --check-prefix=LOONGARCH64_LLVM

// LOONGARCH64: func.func private @cfunc32(i32 {llvm.signext}) -> (i32 {llvm.signext}) attributes {fir.bindc_name = "cfunc32"}

// LOONGARCH64_LLVM: declare signext i32 @cfunc32(i32 signext)
func.func private @cfunc32(i32) -> i32 attributes {fir.bindc_name = "cfunc32"}

// LOONGARCH64-LABEL: func.func @foo(
// LOONGARCH64-SAME: %[[VAL_0:.*]]: i32 {llvm.signext}) -> (i32 {llvm.signext}) attributes {fir.bindc_name = "foo"} {
// LOONGARCH64: %[[VAL_1:.*]] = fir.call @cfunc32(%[[VAL_0]]) fastmath<contract> : (i32) -> i32
// LOONGARCH64: return %[[VAL_1]] : i32
// LOONGARCH64: }

// LOONGARCH64_LLVM-LABEL: define signext i32 @foo(
// LOONGARCH64_LLVM: i32 signext %[[VAL_0:.*]]) {
// LOONGARCH64_LLVM: %[[VAL_1:.*]] = call i32 @cfunc32(i32 %[[VAL_0]])
// LOONGARCH64_LLVM: ret i32 %[[VAL_1]]
// LOONGARCH64_LLVM: }
func.func @foo(%0: i32) -> i32 attributes {fir.bindc_name = "foo"} {
  %1 = fir.call @cfunc32(%0) fastmath<contract> : (i32) -> i32
  return %1 : i32
}
