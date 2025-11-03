// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

typedef unsigned long size_t;

// CIR-LABEL: @_Z4testPc
// LLVM-LABEL: define {{.*}} i64 @_Z4testPc
// OGCG-LABEL: define {{.*}} i64 @_Z4testPc
size_t test(char *ptr) {
  // CIR: cir.objsize max {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false)
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false)
  return __builtin_object_size(ptr, 0);
}

// CIR-LABEL: @_Z8test_minPc
// LLVM-LABEL: define {{.*}} i64 @_Z8test_minPc
// OGCG-LABEL: define {{.*}} i64 @_Z8test_minPc
size_t test_min(char *ptr) {
  // CIR: cir.objsize min {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 false)
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 false)
  return __builtin_object_size(ptr, 2);
}

// CIR-LABEL: @_Z17test_dynamic_sizePc
// LLVM-LABEL: define {{.*}} i64 @_Z17test_dynamic_sizePc
// OGCG-LABEL: define {{.*}} i64 @_Z17test_dynamic_sizePc
size_t test_dynamic_size(char *ptr) {
  // CIR: cir.objsize max dynamic {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 true)
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 true)
  return __builtin_dynamic_object_size(ptr, 0);
}
