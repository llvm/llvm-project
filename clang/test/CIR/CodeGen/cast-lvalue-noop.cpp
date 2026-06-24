// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// CK_NoOp lvalue cast: array of pointers-to-complete-array bound to a
// reference whose element pointer type points to an incomplete array.
void test_noop_array_ptr_to_incomplete() {
  int (*p[2])[4];
  int (*const(&r)[2])[] = p;
  (void)r;
}

// CIR-LABEL: cir.func {{.*}} @_Z33test_noop_array_ptr_to_incompletev
// CIR:         %[[P:.*]] = cir.alloca !cir.array<!cir.ptr<!cir.array<!s32i x 4>> x 2>
// CIR:         %[[R:.*]] = cir.alloca !cir.ptr<!cir.array<!cir.ptr<!cir.array<!s32i x 0>> x 2>>
// CIR:         %[[CAST:.*]] = cir.cast bitcast %[[P]] : !cir.ptr<!cir.array<!cir.ptr<!cir.array<!s32i x 4>> x 2>> -> !cir.ptr<!cir.array<!cir.ptr<!cir.array<!s32i x 0>> x 2>>
// CIR:         cir.store {{.*}} %[[CAST]], %[[R]]

// LLVM-LABEL: define {{.*}} @_Z33test_noop_array_ptr_to_incompletev
// LLVM:         %[[P:.*]] = alloca [2 x ptr]
// LLVM:         %[[R:.*]] = alloca ptr
// LLVM:         store ptr %[[P]], ptr %[[R]]

// OGCG-LABEL: define {{.*}} @_Z33test_noop_array_ptr_to_incompletev
// OGCG:         %[[P:.*]] = alloca [2 x ptr]
// OGCG:         %[[R:.*]] = alloca ptr
// OGCG:         store ptr %[[P]], ptr %[[R]]

// CK_NoOp lvalue cast: double pointer through which the pointee array
// size changes (complete to incomplete) during a qualification conversion.
void test_noop_double_ptr_array() {
  int (*(*p))[4];
  int (*const(*const &r))[] = p;
  (void)r;
}

// CIR-LABEL: cir.func {{.*}} @_Z26test_noop_double_ptr_arrayv
// CIR:         %[[P:.*]] = cir.alloca !cir.ptr<!cir.ptr<!cir.array<!s32i x 4>>>
// CIR:         %[[R:.*]] = cir.alloca !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!s32i x 0>>>>
// CIR:         %[[CAST:.*]] = cir.cast bitcast %[[P]] : !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!s32i x 4>>>> -> !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!s32i x 0>>>>
// CIR:         cir.store {{.*}} %[[CAST]], %[[R]]

// LLVM-LABEL: define {{.*}} @_Z26test_noop_double_ptr_arrayv
// LLVM:         %[[P:.*]] = alloca ptr
// LLVM:         %[[R:.*]] = alloca ptr
// LLVM:         store ptr %[[P]], ptr %[[R]]

// OGCG-LABEL: define {{.*}} @_Z26test_noop_double_ptr_arrayv
// OGCG:         %[[P:.*]] = alloca ptr
// OGCG:         %[[R:.*]] = alloca ptr
// OGCG:         store ptr %[[P]], ptr %[[R]]
