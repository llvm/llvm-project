// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
//
// Should be the same, but sanity check that it doesn't change.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++11 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

int *baseFn();
int idxFn();

// baseFn() is written first, so it must be evaluated before idxFn().
int test1(void) {
  return baseFn()[idxFn()];
}

// CIR-LABEL: cir.func{{.*}} @_Z5test1v
// CIR:         %[[BASE:.*]] = cir.call @_Z6baseFnv()
// CIR-NEXT:    %[[IDX:.*]] = cir.call @_Z5idxFnv()
// CIR:         cir.ptr_stride %[[BASE]], {{.*}}

// LLVM-LABEL: define{{.*}} i32 @_Z5test1v()
// LLVM:         %[[BASE:.*]] = call {{.*}} ptr @_Z6baseFnv()
// LLVM-NEXT:    %[[IDX:.*]] = call {{.*}} i32 @_Z5idxFnv()
// LLVM:         getelementptr {{.*}}i32, ptr %[[BASE]],

int test2(void) {
  return idxFn()[baseFn()];
}

// CIR-LABEL: cir.func{{.*}} @_Z5test2v
// CIR:         %[[IDX:.*]] = cir.call @_Z5idxFnv()
// CIR-NEXT:    %[[BASE:.*]] = cir.call @_Z6baseFnv()
// CIR:         cir.ptr_stride %[[BASE]], {{.*}}

// LLVM-LABEL: define{{.*}} i32 @_Z5test2v()
// LLVM:         %[[IDX:.*]] = call {{.*}} i32 @_Z5idxFnv()
// LLVM-NEXT:    %[[BASE:.*]] = call {{.*}} ptr @_Z6baseFnv()
// LLVM:         getelementptr {{.*}}i32, ptr %[[BASE]],
