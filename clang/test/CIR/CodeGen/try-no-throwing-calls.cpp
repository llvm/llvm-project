// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// Regression test: try block with catch handlers but no throwing calls
// must not crash during TryOp flattening when handler regions reference
// values inlined from the try body.

int nonThrowing() noexcept { return 42; }

int test() {
  int result = 0;
  try {
    result = nonThrowing();
  } catch (...) {
    result = -1;
  }
  return result;
}

// Verify the try body (call + store) is preserved, not discarded.

// CIR: cir.func {{.*}} @_Z4testv
// CIR:   %[[RESULT:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["result", init]
// CIR:   %[[CALL:.*]] = cir.call @_Z11nonThrowingv()
// CIR:   cir.store {{.*}}%[[CALL]], %[[RESULT]]

// LLVM: define {{.*}} @_Z4testv
// LLVM:   %[[CALL:.*]] = call noundef i32 @_Z11nonThrowingv()
// LLVM:   store i32 %[[CALL]], ptr %{{.*}}

// OGCG: define {{.*}} @_Z4testv
// OGCG:   %[[CALL:.*]] = call noundef i32 @_Z11nonThrowingv()
// OGCG:   store i32 %[[CALL]], ptr %{{.*}}
