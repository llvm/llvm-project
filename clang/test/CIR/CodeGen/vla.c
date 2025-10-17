// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
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

// LLVM: define{{.*}} void @f0(i32 %[[LEN_ARG:.*]])
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
// CIR:   %[[SIXTEEN:.*]] = cir.const #cir.int<16> : !s32i
// CIR:   %[[SIXTEEN_SIZE_T:.*]] = cir.cast integral %[[SIXTEEN]] : !s32i -> !u64i
// CIR:   %[[LEN:.*]] = cir.load{{.*}} %[[LEN_ADDR]]
// CIR:   %[[LEN_SIZE_T:.*]] = cir.cast integral %[[LEN]] : !s32i -> !u64i
// CIR:   %[[STACK_PTR:.*]] = cir.stacksave
// CIR:   cir.store{{.*}} %[[STACK_PTR]], %[[SAVED_STACK]]
// CIR:   %[[TOTAL_LEN:.*]] = cir.binop(mul, %[[SIXTEEN_SIZE_T]], %[[LEN_SIZE_T]]) nuw
// CIR:   %[[ARR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, %[[TOTAL_LEN]] : !u64i, ["arr"]
// CIR:   %[[STACK_RESTORE_PTR:.*]] = cir.load{{.*}} %[[SAVED_STACK]]
// CIR:   cir.stackrestore %[[STACK_RESTORE_PTR]]

// LLVM: define{{.*}} void @f1(i32 %[[LEN_ARG:.*]])
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
// CIR:   %[[TOTAL_LEN:.*]] = cir.binop(add, %[[LEN]], %[[FOUR]]) nsw : !s32i
// CIR:   %[[TOTAL_LEN_SIZE_T:.*]] = cir.cast integral %[[TOTAL_LEN]] : !s32i -> !u64i
// CIR:   %[[STACK_PTR:.*]] = cir.stacksave
// CIR:   cir.store{{.*}} %[[STACK_PTR]], %[[SAVED_STACK]]
// CIR:   %[[ARR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, %[[TOTAL_LEN_SIZE_T]] : !u64i, ["arr"]
// CIR:   %[[STACK_RESTORE_PTR:.*]] = cir.load{{.*}} %[[SAVED_STACK]]
// CIR:   cir.stackrestore %[[STACK_RESTORE_PTR]]
  
// LLVM: define{{.*}} void @f2(i32 %[[LEN_ARG:.*]])
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
// CIR:   cir.store{{.*}} %[[LEN_ARG]], %[[LEN_ADDR]]
// CIR:   %[[LEN:.*]] = cir.load{{.*}} %[[LEN_ADDR]]
// CIR:   %[[LEN_SIZE_T:.*]] = cir.cast integral %[[LEN]] : !u32i -> !u64i
// CIR:   %[[STACK_PTR:.*]] = cir.stacksave
// CIR:   cir.store{{.*}} %[[STACK_PTR]], %[[SAVED_STACK]]
// CIR:   %[[S1:.*]] = cir.alloca !s8i, !cir.ptr<!s8i>, %[[LEN_SIZE_T]] : !u64i, ["s1"]
// CIR:   %[[I:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["i", init]
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !u32i
// CIR:   cir.store{{.*}} %[[ZERO]], %[[I]]
// CIR:   cir.scope {
// CIR:     cir.while {
// CIR:     %[[CUR_I:.*]] = cir.load{{.*}} %[[I]]
// CIR:     %[[NEXT:.*]] = cir.unary(inc, %[[CUR_I]])
// CIR:     cir.store{{.*}} %[[NEXT]], %[[I]]
// CIR:     %[[LEN2:.*]] = cir.load{{.*}} %[[LEN_ADDR]]
// CIR:     %[[CMP:.*]] = cir.cmp(lt, %[[NEXT]], %[[LEN2]])
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

// LLVM: define{{.*}} void @f3(i32 %[[LEN_ARG:.*]])
// LLVM:   %[[SAVED_STACK2:.*]] = alloca ptr
// LLVM:   %[[LEN_ADDR:.*]] = alloca i32
// LLVM:   %[[SAVED_STACK:.*]] = alloca ptr
// LLVM:   store i32 %[[LEN_ARG]], ptr %[[LEN_ADDR]]
// LLVM:   %[[LEN:.*]] = load i32, ptr %[[LEN_ADDR]]
// LLVM:   %[[LEN_SIZE_T:.*]] = zext i32 %[[LEN]] to i64
// LLVM:   %[[STACK_PTR:.*]] = call ptr @llvm.stacksave.p0()
// LLVM:   store ptr %[[STACK_PTR]], ptr %[[SAVED_STACK]]
// LLVM:   %[[S1:.*]] = alloca i8, i64 %[[LEN_SIZE_T]]
// LLVM:   %[[I:.*]] = alloca i32
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
  