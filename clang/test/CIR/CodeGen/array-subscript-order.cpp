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
int (&getArr())[4];
void touch();

typedef int v4si __attribute__((vector_size(16)));
v4si &getVec();

typedef int int4 __attribute__((ext_vector_type(4)));
int4 getVec4();

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

int test3(void) {
  return getArr()[idxFn()];
}

// CIR-LABEL: cir.func{{.*}} @_Z5test3v
// CIR:         %[[BASE:.*]] = cir.call @_Z6getArrv()
// CIR-NEXT:    %[[IDX:.*]] = cir.call @_Z5idxFnv()
// CIR:         cir.get_element %[[BASE]][{{.*}}]

// LLVM-LABEL: define{{.*}} i32 @_Z5test3v()
// LLVM:         %[[BASE:.*]] = call {{.*}} ptr @_Z6getArrv()
// LLVM-NEXT:    %[[IDX:.*]] = call {{.*}} i32 @_Z5idxFnv()
// LLVM:         getelementptr {{.*}}[4 x i32], ptr %[[BASE]],

int test_vec(void) {
  return getVec()[idxFn()];
}

// CIR-LABEL: cir.func{{.*}} @_Z8test_vecv
// CIR:         %[[BASE:.*]] = cir.call @_Z6getVecv()
// CIR-NEXT:    %[[VEC:.*]] = cir.load{{.*}}%[[BASE]]
// CIR-NEXT:    %[[IDX:.*]] = cir.call @_Z5idxFnv()
// CIR:         cir.vec.extract %[[VEC]][%[[IDX]]

// LLVM-LABEL: define{{.*}} i32 @_Z8test_vecv()
// LLVM:         %[[BASE:.*]] = call {{.*}} ptr @_Z6getVecv()
// LLVM-NEXT:    %[[VEC:.*]] = load <4 x i32>, ptr %[[BASE]]
// LLVM-NEXT:    %[[IDX:.*]] = call {{.*}} i32 @_Z5idxFnv()
// LLVM:         extractelement <4 x i32> %[[VEC]], i32 %[[IDX]]

int test_extvec(void) {
  return getVec4().xy[idxFn()];
}

// CIR-LABEL: cir.func{{.*}} @_Z11test_extvecv
// CIR:         %[[BASE:.*]] = cir.call @_Z7getVec4v()
// CIR:         %[[IDX:.*]] = cir.call @_Z5idxFnv()
// CIR:         cir.vec.extract {{.*}}[%[[IDX]]

// LLVM-LABEL: define{{.*}} i32 @_Z11test_extvecv()
// LLVM:         %[[BASE:.*]] = call {{.*}} <4 x i32> @_Z7getVec4v()
// LLVM:         %[[IDX:.*]] = call {{.*}} i32 @_Z5idxFnv()
// LLVM:         extractelement {{.*}}, i32 %[[IDX]]

int test_vla(int n, int m, int arr[n][m]) {
  return (touch(), arr)[idxFn()][0];
}

// CIR-LABEL: cir.func{{.*}} @_Z8test_vlaiiPA_i
// CIR:         cir.call @_Z5touchv()
// CIR-NEXT:    %[[ARR:.*]] = cir.load
// CIR-NEXT:    %[[IDX:.*]] = cir.call @_Z5idxFnv()
// CIR:         cir.mul nsw {{.*}}
// CIR:         cir.ptr_stride %[[ARR]], {{.*}}

// LLVM-LABEL: define{{.*}} i32 @_Z8test_vlaiiPA_i
// LLVM:         call void @_Z5touchv()
// LLVM-NEXT:    %[[ARR:.*]] = load ptr, ptr
// LLVM-NEXT:    %[[IDX:.*]] = call {{.*}} i32 @_Z5idxFnv()
// LLVM:         mul nsw i64
// LLVM:         getelementptr {{.*}}ptr %[[ARR]],
