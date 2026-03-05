// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// Test address space conversion via emitPointerWithAlignment (array subscript path).
// This exercises the CK_AddressSpaceConversion case in emitPointerWithAlignment
// where an explicit cast to a different address space is followed by an array
// subscript operation.

#define AS1 __attribute__((address_space(1)))
#define AS2 __attribute__((address_space(2)))

// CIR-LABEL: @_Z24test_cast_then_subscriptPU3AS1i
// LLVM-LABEL: @_Z24test_cast_then_subscriptPU3AS1i
// OGCG-LABEL: @_Z24test_cast_then_subscriptPU3AS1i
void test_cast_then_subscript(AS1 int *p1) {
  // Explicit cast to AS2, then subscript - this goes through emitPointerWithAlignment
  int val = ((AS2 int *)p1)[0];
  // CIR:      %[[#LOAD:]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!s32i, target_address_space(1)>>, !cir.ptr<!s32i, target_address_space(1)>
  // CIR-NEXT: %[[#CAST:]] = cir.cast address_space %[[#LOAD]] : !cir.ptr<!s32i, target_address_space(1)> -> !cir.ptr<!s32i, target_address_space(2)>
  // CIR-NEXT: %[[#IDX:]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: %[[#PTR:]] = cir.ptr_stride %[[#CAST]], %[[#IDX]] : (!cir.ptr<!s32i, target_address_space(2)>, !s32i) -> !cir.ptr<!s32i, target_address_space(2)>
  // CIR-NEXT: %{{.+}} = cir.load {{.*}} %[[#PTR]] : !cir.ptr<!s32i, target_address_space(2)>, !s32i

  // LLVM:      %[[#LOAD:]] = load ptr addrspace(1), ptr %{{.+}}, align 8
  // LLVM-NEXT: %[[#CAST:]] = addrspacecast ptr addrspace(1) %[[#LOAD]] to ptr addrspace(2)
  // LLVM-NEXT: %[[#GEP:]] = getelementptr i32, ptr addrspace(2) %[[#CAST]], i64 0
  // LLVM-NEXT: %{{.+}} = load i32, ptr addrspace(2) %[[#GEP]], align 4

  // OGCG:      %[[#LOAD:]] = load ptr addrspace(1), ptr %{{.+}}, align 8
  // OGCG-NEXT: %[[#CAST:]] = addrspacecast ptr addrspace(1) %[[#LOAD]] to ptr addrspace(2)
  // OGCG:      getelementptr inbounds i32, ptr addrspace(2) %[[#CAST]], i64 0
}

// CIR-LABEL: @_Z30test_cast_then_subscript_writePU3AS1ii
// LLVM-LABEL: @_Z30test_cast_then_subscript_writePU3AS1ii
// OGCG-LABEL: @_Z30test_cast_then_subscript_writePU3AS1ii
void test_cast_then_subscript_write(AS1 int *p1, int val) {
  // Explicit cast to AS2, then subscript for write
  ((AS2 int *)p1)[0] = val;
  // CIR:      %[[#CAST:]] = cir.cast address_space %{{.+}} : !cir.ptr<!s32i, target_address_space(1)> -> !cir.ptr<!s32i, target_address_space(2)>
  // CIR:      %[[#PTR:]] = cir.ptr_stride %[[#CAST]], %{{.+}} : (!cir.ptr<!s32i, target_address_space(2)>, !s32i) -> !cir.ptr<!s32i, target_address_space(2)>
  // CIR-NEXT: cir.store {{.*}}, %[[#PTR]] : !s32i, !cir.ptr<!s32i, target_address_space(2)>

  // LLVM:      %[[#CAST:]] = addrspacecast ptr addrspace(1) %{{.+}} to ptr addrspace(2)
  // LLVM-NEXT: %[[#GEP:]] = getelementptr i32, ptr addrspace(2) %[[#CAST]], i64 0
  // LLVM-NEXT: store i32 %{{.+}}, ptr addrspace(2) %[[#GEP]], align 4

  // OGCG:      %[[#CAST:]] = addrspacecast ptr addrspace(1) %{{.+}} to ptr addrspace(2)
  // OGCG:      getelementptr inbounds i32, ptr addrspace(2) %[[#CAST]], i64 0
}

// CIR-LABEL: @_Z38test_cast_then_subscript_nonzero_indexPU3AS1i
// LLVM-LABEL: @_Z38test_cast_then_subscript_nonzero_indexPU3AS1i
// OGCG-LABEL: @_Z38test_cast_then_subscript_nonzero_indexPU3AS1i
void test_cast_then_subscript_nonzero_index(AS1 int *p1) {
  // Cast then subscript with non-zero index
  int val = ((AS2 int *)p1)[5];
  // CIR:      %[[#CAST:]] = cir.cast address_space %{{.+}} : !cir.ptr<!s32i, target_address_space(1)> -> !cir.ptr<!s32i, target_address_space(2)>
  // CIR:      %[[#IDX:]] = cir.const #cir.int<5> : !s32i
  // CIR-NEXT: %[[#PTR:]] = cir.ptr_stride %[[#CAST]], %[[#IDX]] : (!cir.ptr<!s32i, target_address_space(2)>, !s32i) -> !cir.ptr<!s32i, target_address_space(2)>
  // CIR-NEXT: %{{.+}} = cir.load {{.*}} %[[#PTR]] : !cir.ptr<!s32i, target_address_space(2)>, !s32i

  // LLVM:      %[[#CAST:]] = addrspacecast ptr addrspace(1) %{{.+}} to ptr addrspace(2)
  // LLVM:      %[[#GEP:]] = getelementptr i32, ptr addrspace(2) %[[#CAST]], i64 5
  // LLVM-NEXT: %{{.+}} = load i32, ptr addrspace(2) %[[#GEP]], align 4

  // OGCG:      %[[#CAST:]] = addrspacecast ptr addrspace(1) %{{.+}} to ptr addrspace(2)
  // OGCG:      getelementptr inbounds i32, ptr addrspace(2) %[[#CAST]], i64 5
}
