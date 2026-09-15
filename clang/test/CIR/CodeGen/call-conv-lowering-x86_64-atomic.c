// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// This test relies on atomic type modeling, which is not yet implemented in CIR.
// XFAIL: *

struct AtomicFloats {
  _Atomic(float) a, b;
};

void take_atomic_float(struct AtomicFloats s) {}

// CIR: cir.func {{.*}}@take_atomic_float(
// CIR-SAME: %arg0: !cir.ptr<!rec_AtomicFloats>
// CIR-SAME: llvm.align = 8 : i64
// CIR-SAME: llvm.byval = !rec_AtomicFloats
// LLVM-LABEL: define dso_local void @take_atomic_float(
// LLVM-SAME: ptr noundef byval(%struct.AtomicFloats) align 8 %{{.*}})
