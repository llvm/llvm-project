// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

typedef int vi4 __attribute__((vector_size(16)));

struct SimdStorage {
  int __data __attribute__((vector_size(16)));
};

int scalar_add(int a, int b) { return a + b; }

void vector_binops(vi4 a, vi4 b, SimdStorage &s) {
  vi4 c = a + b;
  vi4 d = a - b;
  vi4 e = a * b;
  s.__data = s.__data + 1;
  s.__data = s.__data - 1;
}

// Scalar signed add keeps nsw.
// CIR: cir.add nsw
// LLVM: add nsw i32

// Vector integer binops must not use nsw/nuw (CIR verifier rejects them).
// CIR: cir.add %{{.*}}, %{{.*}} : !cir.vector<4 x !s32i>
// CIR-NOT: cir.add{{.*}}nsw{{.*}}!cir.vector
// CIR: cir.sub %{{.*}}, %{{.*}} : !cir.vector<4 x !s32i>
// CIR-NOT: cir.sub{{.*}}nsw{{.*}}!cir.vector
// CIR: cir.mul %{{.*}}, %{{.*}} : !cir.vector<4 x !s32i>
// CIR-NOT: cir.mul{{.*}}nsw{{.*}}!cir.vector

// LLVM: add <4 x i32>
// LLVM-NOT: add nsw <4 x i32>
// LLVM: sub <4 x i32>
// LLVM-NOT: sub nsw <4 x i32>
// LLVM: mul <4 x i32>
// LLVM-NOT: mul nsw <4 x i32>
