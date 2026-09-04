// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// Test pointer arithmetic on VLA types.

void test_vla_ptr_add(int n, int i) {
  int arr[n];
  int (*p)[n] = &arr;
  p = p + i;
}

// CIR-LABEL: @test_vla_ptr_add
// CIR:         cir.alloca "arr"
// CIR:         %[[N2:.*]] = cir.load{{.*}} !cir.ptr<!s32i>, !s32i
// CIR:         %[[VLA_SIZE:.*]] = cir.cast integral %[[N2]] : !s32i -> !u64i
// CIR:         %[[P:.*]] = cir.load{{.*}} !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR:         %[[I:.*]] = cir.load{{.*}} !cir.ptr<!s32i>, !s32i
// CIR:         %[[I_EXT:.*]] = cir.cast integral %[[I]] : !s32i -> !u64i
// CIR:         %[[SCALED:.*]] = cir.mul nsw %[[I_EXT]], %[[VLA_SIZE]] : !u64i
// CIR:         cir.ptr_stride %[[P]], %[[SCALED]] : (!cir.ptr<!s32i>, !u64i) -> !cir.ptr<!s32i>

// LLVM-LABEL: @test_vla_ptr_add
// LLVM:         %[[SCALED:.*]] = mul nsw i64 %{{.*}}, %{{.*}}
// LLVM:         getelementptr i32, ptr %{{.*}}, i64 %[[SCALED]]

// OGCG-LABEL: @test_vla_ptr_add
// OGCG:         %[[IDX:.*]] = mul nsw i64 %{{.*}}, %{{.*}}
// OGCG:         getelementptr inbounds i32, ptr %{{.*}}, i64 %[[IDX]]

void test_vla_ptr_inc(int n) {
  int arr[n];
  int (*p)[n] = &arr;
  p++;
}

// CIR-LABEL: @test_vla_ptr_inc
// CIR:         cir.alloca "arr"
// CIR:         %[[N2:.*]] = cir.load{{.*}} !cir.ptr<!s32i>, !s32i
// CIR:         %[[VLA_SIZE:.*]] = cir.cast integral %[[N2]] : !s32i -> !u64i
// CIR:         %[[P:.*]] = cir.load{{.*}} !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR:         cir.ptr_stride %[[P]], %[[VLA_SIZE]] : (!cir.ptr<!s32i>, !u64i) -> !cir.ptr<!s32i>

// LLVM-LABEL: @test_vla_ptr_inc
// LLVM:         getelementptr i32, ptr %{{.*}}, i64 %{{.*}}

// OGCG-LABEL: @test_vla_ptr_inc
// OGCG:         getelementptr inbounds nuw i32, ptr %{{.*}}, i64 %{{.*}}

void test_vla_ptr_dec(int n) {
  int arr[n];
  int (*p)[n] = &arr;
  p--;
}

// CIR-LABEL: @test_vla_ptr_dec
// CIR:         cir.alloca "arr"
// CIR:         %[[N2:.*]] = cir.load{{.*}} !cir.ptr<!s32i>, !s32i
// CIR:         %[[VLA_SIZE:.*]] = cir.cast integral %[[N2]] : !s32i -> !u64i
// CIR:         %[[P:.*]] = cir.load{{.*}} !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR:         %[[SIGNED:.*]] = cir.cast integral %[[VLA_SIZE]] : !u64i -> !s64i
// CIR:         %[[NEG:.*]] = cir.minus nsw %[[SIGNED]] : !s64i
// CIR:         cir.ptr_stride %[[P]], %[[NEG]] : (!cir.ptr<!s32i>, !s64i) -> !cir.ptr<!s32i>

// LLVM-LABEL: @test_vla_ptr_dec
// LLVM:         %[[NEG:.*]] = sub nsw i64 0, %{{.*}}
// LLVM:         getelementptr i32, ptr %{{.*}}, i64 %[[NEG]]

// OGCG-LABEL: @test_vla_ptr_dec
// OGCG:         %[[NEG:.*]] = sub nsw i64 0, %{{.*}}
// OGCG:         getelementptr inbounds i32, ptr %{{.*}}, i64 %[[NEG]]

