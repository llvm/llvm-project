// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

union EmptyUnion {
  EmptyUnion() = default;
};

void f0() {
  EmptyUnion e;
};

// CIR: !rec_EmptyUnion = !cir.record<union "EmptyUnion" padded {!u8i}>
// CIR: cir.func {{.*}} @_Z2f0v()
// CIR:   %0 = cir.alloca !rec_EmptyUnion, !cir.ptr<!rec_EmptyUnion>, ["e"] {alignment = 1 : i64}
// CIR:   cir.return

// LLVM: %union.EmptyUnion = type { i8 }
// LLVM: define dso_local void @_Z2f0v()
// LLVM:   %1 = alloca %union.EmptyUnion, i64 1, align 1
// LLVM:   ret void
