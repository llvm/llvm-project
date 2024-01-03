// Test COMPLEX(10) passing and returning on X86
// REQUIRES: x86-registered-target
// RUN: fir-opt --target-rewrite="target=x86_64-unknown-linux-gnu" %s | FileCheck %s --check-prefix=AMD64
// RUN: tco -target="x86_64-unknown-linux-gnu" %s | FileCheck %s --check-prefix=AMD64_LLVM

func.func @returncomplex10() -> !fir.complex<10> {
  %1 = fir.zero_bits !fir.complex<10>
  return %1 : !fir.complex<10>
}
// AMD64-LABEL:   func.func @returncomplex10() -> tuple<!fir.real<10>, !fir.real<10>> {
// AMD64:    %[[VAL_0:.*]] = fir.zero_bits !fir.complex<10>
// AMD64:    %[[VAL_1:.*]] = fir.alloca tuple<!fir.real<10>, !fir.real<10>>
// AMD64:    %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<tuple<!fir.real<10>, !fir.real<10>>>) -> !fir.ref<!fir.complex<10>>
// AMD64:    fir.store %[[VAL_0]] to %[[VAL_2]] : !fir.ref<!fir.complex<10>>
// AMD64:    %[[VAL_3:.*]] = fir.load %[[VAL_1]] : !fir.ref<tuple<!fir.real<10>, !fir.real<10>>>
// AMD64:    return %[[VAL_3]] : tuple<!fir.real<10>, !fir.real<10>>

// AMD64_LLVM: define { x86_fp80, x86_fp80 } @returncomplex10()

func.func @takecomplex10(%z: !fir.complex<10>) {
  %0 = fir.alloca !fir.complex<10>
  fir.store %z to %0 : !fir.ref<!fir.complex<10>>
  return
}
// AMD64-LABEL:   func.func @takecomplex10(
// AMD64-SAME:  %[[VAL_0:.*]]: !fir.ref<tuple<!fir.real<10>, !fir.real<10>>> {llvm.align = 16 : i32, llvm.byval = tuple<!fir.real<10>, !fir.real<10>>}) {
// AMD64:    %[[VAL_1:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<tuple<!fir.real<10>, !fir.real<10>>>) -> !fir.ref<!fir.complex<10>>
// AMD64:    %[[VAL_2:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.complex<10>>
// AMD64:    %[[VAL_3:.*]] = fir.alloca !fir.complex<10>
// AMD64:    fir.store %[[VAL_2]] to %[[VAL_3]] : !fir.ref<!fir.complex<10>>

// AMD64_LLVM: define void @takecomplex10(ptr byval({ x86_fp80, x86_fp80 }) align 16 %0)
