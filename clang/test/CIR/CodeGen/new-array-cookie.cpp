// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir \
// RUN:   -std=c++14 -fcxx-exceptions -fexceptions %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm \
// RUN:   -std=c++14 -fcxx-exceptions -fexceptions %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm \
// RUN:   -std=c++14 -fcxx-exceptions -fexceptions %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct S {
  ~S();
  int x;
};

// Array new with a non-trivial destructor requires a cookie to store
// the element count.  The cookie is 8 bytes (sizeof(size_t)) and the
// data pointer must be advanced by 8 bytes past the allocation, not
// by 8 * sizeof(ptr).

S *allocArray(int n) {
  return new S[n];
}

// CIR-LABEL: @_Z10allocArrayi
// CIR:   %[[ALLOC:.*]] = cir.call @_Znam
// CIR:   %[[BYTE_PTR:.*]] = cir.cast bitcast %[[ALLOC]] : !cir.ptr<!void> -> !cir.ptr<!u8i>
// CIR:   %[[COOKIE_CAST:.*]] = cir.cast bitcast %[[BYTE_PTR]] : !cir.ptr<!u8i> -> !cir.ptr<!u64i>
// CIR:   cir.store {{.*}} %[[COOKIE_CAST]] : !u64i, !cir.ptr<!u64i>
// CIR:   %[[EIGHT:.*]] = cir.const #cir.int<8> : !s32i
// CIR:   %[[DATA_PTR:.*]] = cir.ptr_stride %[[BYTE_PTR]], %[[EIGHT]] : (!cir.ptr<!u8i>, !s32i) -> !cir.ptr<!u8i>

// LLVM-LABEL: @_Z10allocArrayi
// LLVM:   %[[ALLOC:.*]] = call {{.*}} ptr @_Znam
// LLVM:   store i64 %{{.*}}, ptr %[[ALLOC]], align 8
// LLVM:   %[[DATA:.*]] = getelementptr i8, ptr %[[ALLOC]], i64 8

// OGCG-LABEL: @_Z10allocArrayi
// OGCG:   %[[ALLOC:.*]] = call {{.*}} ptr @_Znam
// OGCG:   store i64 %{{.*}}, ptr %[[ALLOC]], align 8
// OGCG:   %[[DATA:.*]] = getelementptr inbounds i8, ptr %[[ALLOC]], i64 8
