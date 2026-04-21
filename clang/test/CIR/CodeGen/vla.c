// RUN: %clang_cc1 -Wno-error=incompatible-pointer-types -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -Wno-error=incompatible-pointer-types -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -Wno-error=incompatible-pointer-types -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void f0(int len) {
  int arr[len];
}

// CIR: cir.func{{.*}} @f0(%[[LEN_ARG:.*]]: !s32i {{.*}})
// CIR:   %[[LEN_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["len", init]
// CIR:   %[[SAVED_STACK:.*]] = cir.alloca !cir.ptr<!u8i>, !cir.ptr<!cir.ptr<!u8i>>, ["saved_stack"]
// CIR:   cir.store{{.*}} %[[LEN_ARG]], %[[LEN_ADDR]]
// CIR:   %[[LEN:.*]] = cir.load{{.*}} %[[LEN_ADDR]]
// CIR:   %[[LEN_SIZE_T:.*]] = cir.cast integral %[[LEN]] : !s32i -> !u64i
// CIR:   %[[STACK_PTR:.*]] = cir.stacksave
// CIR:   cir.store{{.*}} %[[STACK_PTR]], %[[SAVED_STACK]]
// CIR:   %[[ARR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, %[[LEN_SIZE_T]] : !u64i, ["arr"]
// CIR:   %[[STACK_RESTORE_PTR:.*]] = cir.load{{.*}} %[[SAVED_STACK]]
// CIR:   cir.stackrestore %[[STACK_RESTORE_PTR]]

// LLVM: define{{.*}} void @f0(i32 {{.*}} %[[LEN_ARG:.*]])
// LLVM:   %[[LEN_ADDR:.*]] = alloca i32
// LLVM:   %[[SAVED_STACK:.*]] = alloca ptr
// LLVM:   store i32 %[[LEN_ARG]], ptr %[[LEN_ADDR]]
// LLVM:   %[[LEN:.*]] = load i32, ptr %[[LEN_ADDR]]
// LLVM:   %[[LEN_SIZE_T:.*]] = sext i32 %[[LEN]] to i64
// LLVM:   %[[STACK_PTR:.*]] = call ptr @llvm.stacksave.p0()
// LLVM:   store ptr %[[STACK_PTR]], ptr %[[SAVED_STACK]]
// LLVM:   %[[ARR:.*]] = alloca i32, i64 %[[LEN_SIZE_T]]
// LLVM:   %[[STACK_RESTORE_PTR:.*]] = load ptr, ptr %[[SAVED_STACK]]
// LLVM:   call void @llvm.stackrestore.p0(ptr %[[STACK_RESTORE_PTR]])

// Note: VLA_EXPR0 below is emitted to capture debug info.

// OGCG: define{{.*}} void @f0(i32 {{.*}} %[[LEN_ARG:.*]])
// OGCG:   %[[LEN_ADDR:.*]] = alloca i32
// OGCG:   %[[SAVED_STACK:.*]] = alloca ptr
// OGCG:   %[[VLA_EXPR0:.*]] = alloca i64
// OGCG:   store i32 %[[LEN_ARG]], ptr %[[LEN_ADDR]]
// OGCG:   %[[LEN:.*]] = load i32, ptr %[[LEN_ADDR]]
// OGCG:   %[[LEN_SIZE_T:.*]] = zext i32 %[[LEN]] to i64
// OGCG:   %[[STACK_PTR:.*]] = call ptr @llvm.stacksave.p0()
// OGCG:   store ptr %[[STACK_PTR]], ptr %[[SAVED_STACK]]
// OGCG:   %[[ARR:.*]] = alloca i32, i64 %[[LEN_SIZE_T]]
// OGCG:   store i64 %[[LEN_SIZE_T]], ptr %[[VLA_EXPR0]]
// OGCG:   %[[STACK_RESTORE_PTR:.*]] = load ptr, ptr %[[SAVED_STACK]]
// OGCG:   call void @llvm.stackrestore.p0(ptr %[[STACK_RESTORE_PTR]])

void f1(int len) {
  int arr[16][len];
}

// CIR: cir.func{{.*}} @f1(%[[LEN_ARG:.*]]: !s32i {{.*}})
// CIR:   %[[LEN_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["len", init]
// CIR:   %[[SAVED_STACK:.*]] = cir.alloca !cir.ptr<!u8i>, !cir.ptr<!cir.ptr<!u8i>>, ["saved_stack"]
// CIR:   cir.store{{.*}} %[[LEN_ARG]], %[[LEN_ADDR]]
// CIR:   %[[SIXTEEN:.*]] = cir.const #cir.int<16> : !u64i
// CIR:   %[[LEN:.*]] = cir.load{{.*}} %[[LEN_ADDR]]
// CIR:   %[[LEN_SIZE_T:.*]] = cir.cast integral %[[LEN]] : !s32i -> !u64i
// CIR:   %[[STACK_PTR:.*]] = cir.stacksave
// CIR:   cir.store{{.*}} %[[STACK_PTR]], %[[SAVED_STACK]]
// CIR:   %[[TOTAL_LEN:.*]] = cir.mul nuw %[[SIXTEEN]], %[[LEN_SIZE_T]]
// CIR:   %[[ARR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, %[[TOTAL_LEN]] : !u64i, ["arr"]
// CIR:   %[[STACK_RESTORE_PTR:.*]] = cir.load{{.*}} %[[SAVED_STACK]]
// CIR:   cir.stackrestore %[[STACK_RESTORE_PTR]]

// LLVM: define{{.*}} void @f1(i32 {{.*}} %[[LEN_ARG:.*]])
// LLVM:   %[[LEN_ADDR:.*]] = alloca i32
// LLVM:   %[[SAVED_STACK:.*]] = alloca ptr
// LLVM:   store i32 %[[LEN_ARG]], ptr %[[LEN_ADDR]]
// LLVM:   %[[LEN:.*]] = load i32, ptr %[[LEN_ADDR]]
// LLVM:   %[[LEN_SIZE_T:.*]] = sext i32 %[[LEN]] to i64
// LLVM:   %[[STACK_PTR:.*]] = call ptr @llvm.stacksave.p0()
// LLVM:   store ptr %[[STACK_PTR]], ptr %[[SAVED_STACK]]
// LLVM:   %[[TOTAL_LEN:.*]] = mul nuw i64 16, %[[LEN_SIZE_T]]
// LLVM:   %[[ARR:.*]] = alloca i32, i64 %[[TOTAL_LEN]]
// LLVM:   %[[STACK_RESTORE_PTR:.*]] = load ptr, ptr %[[SAVED_STACK]]
// LLVM:   call void @llvm.stackrestore.p0(ptr %[[STACK_RESTORE_PTR]])

// Note: VLA_EXPR0 below is emitted to capture debug info.

// OGCG: define{{.*}} void @f1(i32 {{.*}} %[[LEN_ARG:.*]])
// OGCG:   %[[LEN_ADDR:.*]] = alloca i32
// OGCG:   %[[SAVED_STACK:.*]] = alloca ptr
// OGCG:   %[[VLA_EXPR0:.*]] = alloca i64
// OGCG:   store i32 %[[LEN_ARG]], ptr %[[LEN_ADDR]]
// OGCG:   %[[LEN:.*]] = load i32, ptr %[[LEN_ADDR]]
// OGCG:   %[[LEN_SIZE_T:.*]] = zext i32 %[[LEN]] to i64
// OGCG:   %[[STACK_PTR:.*]] = call ptr @llvm.stacksave.p0()
// OGCG:   store ptr %[[STACK_PTR]], ptr %[[SAVED_STACK]]
// OGCG:   %[[TOTAL_LEN:.*]] = mul nuw i64 16, %[[LEN_SIZE_T]]
// OGCG:   %[[ARR:.*]] = alloca i32, i64 %[[TOTAL_LEN]]
// OGCG:   store i64 %[[LEN_SIZE_T]], ptr %[[VLA_EXPR0]]
// OGCG:   %[[STACK_RESTORE_PTR:.*]] = load ptr, ptr %[[SAVED_STACK]]
// OGCG:   call void @llvm.stackrestore.p0(ptr %[[STACK_RESTORE_PTR]])

void f2(int len) {
  int arr[len + 4];
}
  
// CIR: cir.func{{.*}} @f2(%[[LEN_ARG:.*]]: !s32i {{.*}})
// CIR:   %[[LEN_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["len", init]
// CIR:   %[[SAVED_STACK:.*]] = cir.alloca !cir.ptr<!u8i>, !cir.ptr<!cir.ptr<!u8i>>, ["saved_stack"]
// CIR:   cir.store{{.*}} %[[LEN_ARG]], %[[LEN_ADDR]]
// CIR:   %[[LEN:.*]] = cir.load{{.*}} %[[LEN_ADDR]]
// CIR:   %[[FOUR:.*]] = cir.const #cir.int<4> : !s32i
// CIR:   %[[TOTAL_LEN:.*]] = cir.add nsw %[[LEN]], %[[FOUR]] : !s32i
// CIR:   %[[TOTAL_LEN_SIZE_T:.*]] = cir.cast integral %[[TOTAL_LEN]] : !s32i -> !u64i
// CIR:   %[[STACK_PTR:.*]] = cir.stacksave
// CIR:   cir.store{{.*}} %[[STACK_PTR]], %[[SAVED_STACK]]
// CIR:   %[[ARR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, %[[TOTAL_LEN_SIZE_T]] : !u64i, ["arr"]
// CIR:   %[[STACK_RESTORE_PTR:.*]] = cir.load{{.*}} %[[SAVED_STACK]]
// CIR:   cir.stackrestore %[[STACK_RESTORE_PTR]]
  
// LLVM: define{{.*}} void @f2(i32 {{.*}} %[[LEN_ARG:.*]])
// LLVM:   %[[LEN_ADDR:.*]] = alloca i32
// LLVM:   %[[SAVED_STACK:.*]] = alloca ptr
// LLVM:   store i32 %[[LEN_ARG]], ptr %[[LEN_ADDR]]
// LLVM:   %[[LEN:.*]] = load i32, ptr %[[LEN_ADDR]]
// LLVM:   %[[TOTAL_LEN:.*]] = add nsw i32 %[[LEN]], 4
// LLVM:   %[[TOTAL_LEN_SIZE_T:.*]] = sext i32 %[[TOTAL_LEN]] to i64
// LLVM:   %[[STACK_PTR:.*]] = call ptr @llvm.stacksave.p0()
// LLVM:   store ptr %[[STACK_PTR]], ptr %[[SAVED_STACK]]
// LLVM:   %[[ARR:.*]] = alloca i32, i64 %[[TOTAL_LEN_SIZE_T]]
// LLVM:   %[[STACK_RESTORE_PTR:.*]] = load ptr, ptr %[[SAVED_STACK]]
// LLVM:   call void @llvm.stackrestore.p0(ptr %[[STACK_RESTORE_PTR]])
  
// Note: VLA_EXPR0 below is emitted to capture debug info.
  
// OGCG: define{{.*}} void @f2(i32 {{.*}} %[[LEN_ARG:.*]])
// OGCG:   %[[LEN_ADDR:.*]] = alloca i32
// OGCG:   %[[SAVED_STACK:.*]] = alloca ptr
// OGCG:   %[[VLA_EXPR0:.*]] = alloca i64
// OGCG:   store i32 %[[LEN_ARG]], ptr %[[LEN_ADDR]]
// OGCG:   %[[LEN:.*]] = load i32, ptr %[[LEN_ADDR]]
// OGCG:   %[[TOTAL_LEN:.*]] = add nsw i32 %[[LEN]], 4
// OGCG:   %[[TOTAL_LEN_SIZE_T:.*]] = zext i32 %[[TOTAL_LEN]] to i64
// OGCG:   %[[STACK_PTR:.*]] = call ptr @llvm.stacksave.p0()
// OGCG:   store ptr %[[STACK_PTR]], ptr %[[SAVED_STACK]]
// OGCG:   %[[ARR:.*]] = alloca i32, i64 %[[TOTAL_LEN_SIZE_T]]
// OGCG:   store i64 %[[TOTAL_LEN_SIZE_T]], ptr %[[VLA_EXPR0]]
// OGCG:   %[[STACK_RESTORE_PTR:.*]] = load ptr, ptr %[[SAVED_STACK]]
// OGCG:   call void @llvm.stackrestore.p0(ptr %[[STACK_RESTORE_PTR]])

void f3(unsigned len) {
  char s1[len];
  unsigned i = 0u;
  while (++i < len) {
    char s2[i];
  }
}

// CIR: cir.func{{.*}} @f3(%[[LEN_ARG:.*]]: !u32i {{.*}})
// CIR:   %[[LEN_ADDR:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["len", init]
// CIR:   %[[SAVED_STACK:.*]] = cir.alloca !cir.ptr<!u8i>, !cir.ptr<!cir.ptr<!u8i>>, ["saved_stack"]
// CIR:   %[[I:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["i", init]
// CIR:   cir.store{{.*}} %[[LEN_ARG]], %[[LEN_ADDR]]
// CIR:   %[[LEN:.*]] = cir.load{{.*}} %[[LEN_ADDR]]
// CIR:   %[[LEN_SIZE_T:.*]] = cir.cast integral %[[LEN]] : !u32i -> !u64i
// CIR:   %[[STACK_PTR:.*]] = cir.stacksave
// CIR:   cir.store{{.*}} %[[STACK_PTR]], %[[SAVED_STACK]]
// CIR:   %[[S1:.*]] = cir.alloca !s8i, !cir.ptr<!s8i>, %[[LEN_SIZE_T]] : !u64i, ["s1"]
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !u32i
// CIR:   cir.store{{.*}} %[[ZERO]], %[[I]]
// CIR:   cir.scope {
// CIR:     cir.while {
// CIR:     %[[CUR_I:.*]] = cir.load{{.*}} %[[I]]
// CIR:     %[[NEXT:.*]] = cir.inc %[[CUR_I]]
// CIR:     cir.store{{.*}} %[[NEXT]], %[[I]]
// CIR:     %[[LEN2:.*]] = cir.load{{.*}} %[[LEN_ADDR]]
// CIR:     %[[CMP:.*]] = cir.cmp lt %[[NEXT]], %[[LEN2]]
// CIR:     cir.condition(%[[CMP]])
// CIR:   } do {
// CIR:       cir.scope {
// CIR:         %[[SAVED_STACK2:.*]] = cir.alloca !cir.ptr<!u8i>, !cir.ptr<!cir.ptr<!u8i>>, ["saved_stack"]
// CIR:         %[[I_LEN:.*]] = cir.load{{.*}} %[[I]]
// CIR:         %[[I_LEN_SIZE_T2:.*]] = cir.cast integral %[[I_LEN]] : !u32i -> !u64i
// CIR:         %[[STACK_PTR2:.*]] = cir.stacksave
// CIR:         cir.store{{.*}} %[[STACK_PTR2]], %[[SAVED_STACK2]]
// CIR:         %[[S2:.*]] = cir.alloca !s8i, !cir.ptr<!s8i>, %[[I_LEN_SIZE_T2]] : !u64i, ["s2"]
// CIR:         %[[SAVED_RESTORE_PTR2:.*]] = cir.load{{.*}} %[[SAVED_STACK2]]
// CIR:         cir.stackrestore %[[SAVED_RESTORE_PTR2]]
// CIR:       }
// CIR:       cir.yield
// CIR:     }
// CIR:   }
// CIR:   %[[STACK_RESTORE_PTR:.*]] = cir.load{{.*}} %[[SAVED_STACK]]
// CIR:   cir.stackrestore %[[STACK_RESTORE_PTR]]

// LLVM: define{{.*}} void @f3(i32 {{.*}} %[[LEN_ARG:.*]])
// LLVM:   %[[SAVED_STACK2:.*]] = alloca ptr
// LLVM:   %[[LEN_ADDR:.*]] = alloca i32
// LLVM:   %[[SAVED_STACK:.*]] = alloca ptr
// LLVM:   %[[I:.*]] = alloca i32
// LLVM:   store i32 %[[LEN_ARG]], ptr %[[LEN_ADDR]]
// LLVM:   %[[LEN:.*]] = load i32, ptr %[[LEN_ADDR]]
// LLVM:   %[[LEN_SIZE_T:.*]] = zext i32 %[[LEN]] to i64
// LLVM:   %[[STACK_PTR:.*]] = call ptr @llvm.stacksave.p0()
// LLVM:   store ptr %[[STACK_PTR]], ptr %[[SAVED_STACK]]
// LLVM:   %[[S1:.*]] = alloca i8, i64 %[[LEN_SIZE_T]]
// LLVM:   store i32 0, ptr %[[I]]
// LLVM:   br label %[[WHILE_START:.*]]
// LLVM: [[WHILE_START]]:
// LLVM:   br label %[[WHILE_COND:.*]]
// LLVM: [[WHILE_COND]]:
// LLVM:   %[[CUR_I:.*]] = load i32, ptr %[[I]]
// LLVM:   %[[NEXT:.*]] = add i32 %[[CUR_I]], 1
// LLVM:   store i32 %[[NEXT]], ptr %[[I]]
// LLVM:   %[[LEN2:.*]] = load i32, ptr %[[LEN_ADDR]]
// LLVM:   %[[CMP:.*]] = icmp ult i32 %[[NEXT]], %[[LEN2]]
// LLVM:   br i1 %[[CMP]], label %[[WHILE_BODY:.*]], label %[[WHILE_END:.*]]
// LLVM: [[WHILE_BODY]]:
// LLVM:   br label %[[WHILE_BODY2:.*]]
// LLVM: [[WHILE_BODY2]]:
// LLVM:   %[[I_LEN:.*]] = load i32, ptr %[[I]]
// LLVM:   %[[I_LEN_SIZE_T2:.*]] = zext i32 %[[I_LEN]] to i64
// LLVM:   %[[STACK_PTR2:.*]] = call ptr @llvm.stacksave.p0()
// LLVM:   store ptr %[[STACK_PTR2]], ptr %[[SAVED_STACK2]]
// LLVM:   %[[S2:.*]] = alloca i8, i64 %[[I_LEN_SIZE_T2]]
// LLVM:   %[[STACK_RESTORE_PTR2:.*]] = load ptr, ptr %[[SAVED_STACK2]]
// LLVM:   call void @llvm.stackrestore.p0(ptr %[[STACK_RESTORE_PTR2]])
// LLVM:   br label %[[WHILE_BODY_END:.*]]
// LLVM: [[WHILE_BODY_END]]:
// LLVM:   br label %[[WHILE_COND]]
// LLVM: [[WHILE_END]]:
// LLVM:   br label %[[F3_END:.*]]
// LLVM: [[F3_END]]:
// LLVM:   %[[STACK_RESTORE_PTR:.*]] = load ptr, ptr %[[SAVED_STACK]]
// LLVM:   call void @llvm.stackrestore.p0(ptr %[[STACK_RESTORE_PTR]])

// Note: VLA_EXPR0 and VLA_EXPR1 below are emitted to capture debug info.

// OGCG: define{{.*}} void @f3(i32 {{.*}} %[[LEN_ARG:.*]])
// OGCG:   %[[LEN_ADDR:.*]] = alloca i32
// OGCG:   %[[SAVED_STACK:.*]] = alloca ptr
// OGCG:   %[[VLA_EXPR0:.*]] = alloca i64
// OGCG:   %[[I:.*]] = alloca i32
// OGCG:   %[[SAVED_STACK1:.*]] = alloca ptr
// OGCG:   %[[VLA_EXPR1:.*]] = alloca i64
// OGCG:   store i32 %[[LEN_ARG]], ptr %[[LEN_ADDR]]
// OGCG:   %[[LEN:.*]] = load i32, ptr %[[LEN_ADDR]]
// OGCG:   %[[LEN_SIZE_T:.*]] = zext i32 %[[LEN]] to i64
// OGCG:   %[[STACK_PTR:.*]] = call ptr @llvm.stacksave.p0()
// OGCG:   store ptr %[[STACK_PTR]], ptr %[[SAVED_STACK]]
// OGCG:   %[[S1:.*]] = alloca i8, i64 %[[LEN_SIZE_T]]
// OGCG:   store i64 %[[LEN_SIZE_T]], ptr %[[VLA_EXPR0]]
// OGCG:   br label %[[WHILE_COND:.*]]
// OGCG: [[WHILE_COND]]:
// OGCG:   %[[CUR_I:.*]] = load i32, ptr %[[I]]
// OGCG:   %[[NEXT:.*]] = add i32 %[[CUR_I]], 1
// OGCG:   store i32 %[[NEXT]], ptr %[[I]]
// OGCG:   %[[LEN2:.*]] = load i32, ptr %[[LEN_ADDR]]
// OGCG:   %[[CMP:.*]] = icmp ult i32 %[[NEXT]], %[[LEN2]]
// OGCG:   br i1 %[[CMP]], label %[[WHILE_BODY:.*]], label %[[WHILE_END:.*]]
// OGCG: [[WHILE_BODY]]:
// OGCG:   %[[I_LEN:.*]] = load i32, ptr %[[I]]
// OGCG:   %[[I_LEN_SIZE_T:.*]] = zext i32 %[[I_LEN]] to i64
// OGCG:   %[[STACK_PTR1:.*]] = call ptr @llvm.stacksave.p0()
// OGCG:   store ptr %[[STACK_PTR1]], ptr %[[SAVED_STACK1]]
// OGCG:   %[[S2:.*]] = alloca i8, i64 %[[I_LEN_SIZE_T]]
// OGCG:   store i64 %[[I_LEN_SIZE_T]], ptr %[[VLA_EXPR1]]
// OGCG:   %[[STACK_RESTORE_PTR1:.*]] = load ptr, ptr %[[SAVED_STACK1]]
// OGCG:   call void @llvm.stackrestore.p0(ptr %[[STACK_RESTORE_PTR1]])
// OGCG:   br label %[[WHILE_COND]]
// OGCG: [[WHILE_END]]:
// OGCG:   %[[STACK_RESTORE_PTR:.*]] = load ptr, ptr %[[SAVED_STACK]]
// OGCG:   call void @llvm.stackrestore.p0(ptr %[[STACK_RESTORE_PTR]])


// The following test case is disabled because it runs into a bug (unrelated
// to VLA) in the handling of cleanups in loops with break statements.
//
// void f4(unsigned len) {
//     char s1[len];
//     while (1) {
//       char s2[len];
//       if (1)
//         break;
//     }
// }

int f5(unsigned long len) {
  int arr[len];
  return arr[2];
}

// CIR: cir.func{{.*}} @f5(%[[LEN_ARG:.*]]: !u64i {{.*}}) -> !s32i
// CIR:   %[[LEN_ADDR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["len", init]
// CIR:   %[[RET_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:   %[[SAVED_STACK:.*]] = cir.alloca !cir.ptr<!u8i>, !cir.ptr<!cir.ptr<!u8i>>, ["saved_stack"]
// CIR:   cir.store{{.*}} %[[LEN_ARG]], %[[LEN_ADDR]]
// CIR:   %[[LEN:.*]] = cir.load{{.*}} %[[LEN_ADDR]]
// CIR:   %[[STACK_PTR:.*]] = cir.stacksave
// CIR:   cir.store{{.*}} %[[STACK_PTR]], %[[SAVED_STACK]]
// CIR:   %[[ARR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, %[[LEN]] : !u64i, ["arr"]
// CIR:   %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
// CIR:   %[[ARR_2:.*]] = cir.ptr_stride %[[ARR]], %[[TWO]]
// CIR:   %[[ARR_VAL:.*]] = cir.load{{.*}} %[[ARR_2]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.store{{.*}} %[[ARR_VAL]], %[[RET_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[STACK_RESTORE_PTR:.*]] = cir.load{{.*}} %[[SAVED_STACK]]
// CIR:   cir.stackrestore %[[STACK_RESTORE_PTR]]
// CIR:   %[[RET_VAL:.*]] = cir.load{{.*}} %[[RET_ADDR]]
// CIR:   cir.return %[[RET_VAL]] : !s32i

// LLVM: define{{.*}} i32 @f5(i64 {{.*}} %[[LEN_ARG:.*]])
// LLVM:   %[[LEN_ADDR:.*]] = alloca i64
// LLVM:   %[[RET_ADDR:.*]] = alloca i32
// LLVM:   %[[SAVED_STACK:.*]] = alloca ptr
// LLVM:   store i64 %[[LEN_ARG]], ptr %[[LEN_ADDR]]
// LLVM:   %[[LEN:.*]] = load i64, ptr %[[LEN_ADDR]]
// LLVM:   %[[STACK_PTR:.*]] = call ptr @llvm.stacksave.p0()
// LLVM:   store ptr %[[STACK_PTR]], ptr %[[SAVED_STACK]]
// LLVM:   %[[ARR:.*]] = alloca i32, i64 %[[LEN]]
// LLVM:   %[[ARR_2:.*]] = getelementptr i32, ptr %[[ARR]], i64 2
// LLVM:   %[[ARR_VAL:.*]] = load i32, ptr %[[ARR_2]]
// LLVM:   store i32 %[[ARR_VAL]], ptr %[[RET_ADDR]]
// LLVM:   %[[STACK_RESTORE_PTR:.*]] = load ptr, ptr %[[SAVED_STACK]]
// LLVM:   call void @llvm.stackrestore.p0(ptr %[[STACK_RESTORE_PTR]])
// LLVM:   %[[RET_VAL:.*]] = load i32, ptr %[[RET_ADDR]]
// LLVM:   ret i32 %[[RET_VAL]]

// Note: VLA_EXPR0 below is emitted to capture debug info.

// OGCG: define{{.*}} i32 @f5(i64 {{.*}} %[[LEN_ARG:.*]])
// OGCG:   %[[LEN_ADDR:.*]] = alloca i64
// OGCG:   %[[SAVED_STACK:.*]] = alloca ptr
// OGCG:   %[[VLA_EXPR0:.*]] = alloca i64
// OGCG:   store i64 %[[LEN_ARG]], ptr %[[LEN_ADDR]]
// OGCG:   %[[LEN:.*]] = load i64, ptr %[[LEN_ADDR]]
// OGCG:   %[[STACK_PTR:.*]] = call ptr @llvm.stacksave.p0()
// OGCG:   store ptr %[[STACK_PTR]], ptr %[[SAVED_STACK]]
// OGCG:   %[[ARR:.*]] = alloca i32, i64 %[[LEN]]
// OGCG:   store i64 %[[LEN]], ptr %[[VLA_EXPR0]]
// OGCG:   %[[ARR_2:.*]] = getelementptr inbounds i32, ptr %[[ARR]], i64 2
// OGCG:   %[[ARR_VAL:.*]] = load i32, ptr %[[ARR_2]]
// OGCG:   %[[STACK_RESTORE_PTR:.*]] = load ptr, ptr %[[SAVED_STACK]]
// OGCG:   call void @llvm.stackrestore.p0(ptr %[[STACK_RESTORE_PTR]])
// OGCG:   ret i32 %[[ARR_VAL]]

void vla_subscript_expr() {
  int **a;
  unsigned long n = 5;
  (int (**)[n]){&a}[0][1][5] = 0;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, ["a"]
// CIR: %[[N_ADDR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["n", init]
// CIR: %[[COMPOUND_ADDR:.*]] = cir.alloca !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, [".compoundliteral"]
// CIR: %[[CONST_5:.*]] = cir.const #cir.int<5> : !u64i
// CIR: cir.store {{.*}} %[[CONST_5]], %[[N_ADDR]] : !u64i, !cir.ptr<!u64i>
// CIR: %[[CONST_0_VAL:.*]] = cir.const #cir.int<0> : !s32i
// CIR: %[[CONST_5:.*]] = cir.const #cir.int<5> : !s32i
// CIR: %[[CONST_0:.*]] = cir.const #cir.int<0> : !s32i
// CIR: %[[TMP_N:.*]] = cir.load {{.*}} %[[N_ADDR]] : !cir.ptr<!u64i>, !u64i
// CIR: %[[A_VAL:.*]] = cir.cast bitcast %[[A_ADDR]] : !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>> -> !cir.ptr<!cir.ptr<!s32i>>
// CIR: cir.store {{.*}} %[[A_VAL]], %[[COMPOUND_ADDR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>
// CIR: %[[TMP_COMPOUND:.*]] = cir.load {{.*}} %[[COMPOUND_ADDR]] : !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, !cir.ptr<!cir.ptr<!s32i>>
// CIR: %[[COMPOUND_PTR:.*]] = cir.ptr_stride %[[TMP_COMPOUND]], %[[CONST_0]] : (!cir.ptr<!cir.ptr<!s32i>>, !s32i) -> !cir.ptr<!cir.ptr<!s32i>>
// CIR: %[[TMP_COMPOUND:.*]] = cir.load {{.*}} %[[COMPOUND_PTR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR: %[[CONST_1:.*]] = cir.const #cir.int<1> : !u64i
// CIR: %[[VLA_IDX:.*]] = cir.mul nsw %[[CONST_1]], %[[TMP_N]] : !u64i
// CIR: %[[VLA_A_PTR:.*]] = cir.ptr_stride %[[TMP_COMPOUND]], %[[VLA_IDX]] : (!cir.ptr<!s32i>, !u64i) -> !cir.ptr<!s32i>
// CIR: %[[ELEM_5_PTR:.*]] = cir.ptr_stride %[[VLA_A_PTR]], %[[CONST_5]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
// CIR: cir.store {{.*}} %[[CONST_0_VAL]], %[[ELEM_5_PTR]] : !s32i, !cir.ptr<!s32i>

// LLVM: %[[A_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM: %[[N_ADDR:.*]] = alloca i64, i64 1, align 8
// LLVM: %[[COMPOUND_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM: store i64 5, ptr %[[N_ADDR]], align 8
// LLVM: %[[TMP_N:.*]] = load i64, ptr %[[N_ADDR]], align 8
// LLVM: store ptr %[[A_ADDR]], ptr %[[COMPOUND_ADDR]], align 8
// LLVM: %[[TMP_COMPOUND:.*]] = load ptr, ptr %[[COMPOUND_ADDR]], align 8
// LLVM: %[[COMPOUND_PTR:.*]] = getelementptr ptr, ptr %[[TMP_COMPOUND]], i64 0
// LLVM: %[[TMP_COMPOUND:.*]] = load ptr, ptr %[[COMPOUND_PTR]], align 8
// LLVM: %[[VLA_IDX:.*]] = mul nsw i64 1, %[[TMP_N]]
// LLVM: %[[VLA_A_PTR:.*]] = getelementptr i32, ptr %[[TMP_COMPOUND]], i64 %[[VLA_IDX]]
// LLVM: %[[ELEM_5_PTR:.*]] = getelementptr i32, ptr %[[VLA_A_PTR]], i64 5
// LLVM: store i32 0, ptr %[[ELEM_5_PTR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca ptr, align 8
// OGCG: %[[N_ADDR:.*]] = alloca i64, align 8
// OGCG: %[[COMPOUND_ADDR:.*]] = alloca ptr, align 8
// OGCG: store i64 5, ptr %[[N_ADDR]], align 8
// OGCG: %[[TMP_N:.*]] = load i64, ptr %[[N_ADDR]], align 8
// OGCG: store ptr %[[A_ADDR]], ptr %[[COMPOUND_ADDR]], align 8
// OGCG: %[[TMP_COMPOUND:.*]] = load ptr, ptr %[[COMPOUND_ADDR]], align 8
// OGCG: %[[COMPOUND_PTR:.*]] = getelementptr inbounds ptr, ptr %[[TMP_COMPOUND]], i64 0
// OGCG: %[[TMP_COMPOUND:.*]] = load ptr, ptr %[[COMPOUND_PTR]], align 8
// OGCG: %[[VLA_IDX:.*]] = mul nsw i64 1, %[[TMP_N]]
// OGCG: %[[VLA_A_PTR:.*]] = getelementptr inbounds i32, ptr %[[TMP_COMPOUND]], i64 %[[VLA_IDX]]
// OGCG: %[[ELEM_5_PTR:.*]] = getelementptr inbounds i32, ptr %[[VLA_A_PTR]], i64 5
// OGCG: store i32 0, ptr %[[ELEM_5_PTR]], align 4

double vla_param_2d(int n, double m[n][n], int i, int j) {
  return m[i][j];
}

// CIR: cir.func{{.*}} @vla_param_2d(%[[N_ARG:.*]]: !s32i {{.*}}, %[[M_ARG:.*]]: !cir.ptr<!cir.double> {{.*}}, %[[I_ARG:.*]]: !s32i {{.*}}, %[[J_ARG:.*]]: !s32i {{.*}}) -> !cir.double
// CIR:   %[[N_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["n", init]
// CIR:   %[[M_ADDR:.*]] = cir.alloca !cir.ptr<!cir.double>, !cir.ptr<!cir.ptr<!cir.double>>, ["m", init]
// CIR:   %[[I_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init]
// CIR:   %[[J_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["j", init]
// CIR:   cir.store{{.*}} %[[N_ARG]], %[[N_ADDR]]
// CIR:   cir.store{{.*}} %[[M_ARG]], %[[M_ADDR]]
// CIR:   cir.store{{.*}} %[[I_ARG]], %[[I_ADDR]]
// CIR:   cir.store{{.*}} %[[J_ARG]], %[[J_ADDR]]
// CIR:   %[[N:.*]] = cir.load{{.*}} %[[N_ADDR]]
// CIR:   %[[VLA_SIZE:.*]] = cir.cast integral %[[N]] : !s32i -> !u64i
// CIR:   %[[J:.*]] = cir.load{{.*}} %[[J_ADDR]]
// CIR:   %[[I:.*]] = cir.load{{.*}} %[[I_ADDR]]
// CIR:   %[[M:.*]] = cir.load{{.*}} %[[M_ADDR]]
// CIR:   %[[I_EXT:.*]] = cir.cast integral %[[I]] : !s32i -> !u64i
// CIR:   %[[ROW_OFF:.*]] = cir.mul nsw %[[I_EXT]], %[[VLA_SIZE]] : !u64i
// CIR:   %[[ROW_PTR:.*]] = cir.ptr_stride %[[M]], %[[ROW_OFF]]
// CIR:   %[[ELEM_PTR:.*]] = cir.ptr_stride %[[ROW_PTR]], %[[J]]
// CIR:   %[[ELEM:.*]] = cir.load{{.*}} %[[ELEM_PTR]] : !cir.ptr<!cir.double>, !cir.double

// LLVM: define{{.*}} double @vla_param_2d(i32 {{.*}} %[[N_ARG:.*]], ptr {{.*}} %[[M_ARG:.*]], i32 {{.*}} %[[I_ARG:.*]], i32 {{.*}} %[[J_ARG:.*]])
// LLVM:   %[[N_ADDR:.*]] = alloca i32
// LLVM:   %[[M_ADDR:.*]] = alloca ptr
// LLVM:   %[[I_ADDR:.*]] = alloca i32
// LLVM:   %[[J_ADDR:.*]] = alloca i32
// LLVM:   store i32 %[[N_ARG]], ptr %[[N_ADDR]]
// LLVM:   store ptr %[[M_ARG]], ptr %[[M_ADDR]]
// LLVM:   store i32 %[[I_ARG]], ptr %[[I_ADDR]]
// LLVM:   store i32 %[[J_ARG]], ptr %[[J_ADDR]]
// LLVM:   %[[N:.*]] = load i32, ptr %[[N_ADDR]]
// LLVM:   %[[VLA_SIZE:.*]] = sext i32 %[[N]] to i64
// LLVM:   %[[J:.*]] = load i32, ptr %[[J_ADDR]]
// LLVM:   %[[I:.*]] = load i32, ptr %[[I_ADDR]]
// LLVM:   %[[M:.*]] = load ptr, ptr %[[M_ADDR]]
// LLVM:   %[[I_EXT:.*]] = sext i32 %[[I]] to i64
// LLVM:   %[[ROW_OFF:.*]] = mul nsw i64 %[[I_EXT]], %[[VLA_SIZE]]
// LLVM:   %[[ROW_PTR:.*]] = getelementptr double, ptr %[[M]], i64 %[[ROW_OFF]]
// LLVM:   %[[J_EXT:.*]] = sext i32 %[[J]] to i64
// LLVM:   %[[ELEM_PTR:.*]] = getelementptr double, ptr %[[ROW_PTR]], i64 %[[J_EXT]]
// LLVM:   %[[ELEM:.*]] = load double, ptr %[[ELEM_PTR]]

// OGCG: define{{.*}} double @vla_param_2d(i32 {{.*}} %[[N_ARG:.*]], ptr {{.*}} %[[M_ARG:.*]], i32 {{.*}} %[[I_ARG:.*]], i32 {{.*}} %[[J_ARG:.*]])
// OGCG:   %[[N_ADDR:.*]] = alloca i32
// OGCG:   %[[M_ADDR:.*]] = alloca ptr
// OGCG:   %[[I_ADDR:.*]] = alloca i32
// OGCG:   %[[J_ADDR:.*]] = alloca i32
// OGCG:   store i32 %[[N_ARG]], ptr %[[N_ADDR]]
// OGCG:   store ptr %[[M_ARG]], ptr %[[M_ADDR]]
// OGCG:   store i32 %[[I_ARG]], ptr %[[I_ADDR]]
// OGCG:   store i32 %[[J_ARG]], ptr %[[J_ADDR]]
// OGCG:   %[[N0:.*]] = load i32, ptr %[[N_ADDR]]
// OGCG:   %{{.*}} = zext i32 %[[N0]] to i64
// OGCG:   %[[N1:.*]] = load i32, ptr %[[N_ADDR]]
// OGCG:   %[[VLA_SIZE:.*]] = zext i32 %[[N1]] to i64
// OGCG:   %[[M:.*]] = load ptr, ptr %[[M_ADDR]]
// OGCG:   %[[I:.*]] = load i32, ptr %[[I_ADDR]]
// OGCG:   %[[I_EXT:.*]] = sext i32 %[[I]] to i64
// OGCG:   %[[ROW_OFF:.*]] = mul nsw i64 %[[I_EXT]], %[[VLA_SIZE]]
// OGCG:   %[[ROW_PTR:.*]] = getelementptr inbounds double, ptr %[[M]], i64 %[[ROW_OFF]]
// OGCG:   %[[J:.*]] = load i32, ptr %[[J_ADDR]]
// OGCG:   %[[J_EXT:.*]] = sext i32 %[[J]] to i64
// OGCG:   %[[ELEM_PTR:.*]] = getelementptr inbounds double, ptr %[[ROW_PTR]], i64 %[[J_EXT]]
// OGCG:   %[[ELEM:.*]] = load double, ptr %[[ELEM_PTR]]
