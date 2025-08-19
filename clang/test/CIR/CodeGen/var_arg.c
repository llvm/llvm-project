// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// CIR: !rec___va_list_tag = !cir.record<struct "__va_list_tag" {!u32i, !u32i, !cir.ptr<!void>, !cir.ptr<!void>}
// LLVM: %struct.__va_list_tag = type { i32, i32, ptr, ptr }
// OGCG: %struct.__va_list_tag = type { i32, i32, ptr, ptr }

void varargs(int count, ...) {
    __builtin_va_list args;
    __builtin_va_start(args, 12345);
    __builtin_va_end(args);
}

// CIR-LABEL: cir.func dso_local @varargs
// CIR-SAME: (%[[COUNT:.+]]: !s32i{{.*}}, ...)
// CIR:   %[[COUNT_ADDR:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["count", init]
// CIR:   %[[ARGS:.+]] = cir.alloca !cir.array<!rec___va_list_tag x 1>, !cir.ptr<!cir.array<!rec___va_list_tag x 1>>, ["args"]
// CIR:   cir.store %[[COUNT]], %[[COUNT_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[C12345:.+]] = cir.const #cir.int<12345> : !s32i
// CIR:   %[[APTR:.+]] = cir.cast(array_to_ptrdecay, %[[ARGS]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>>), !cir.ptr<!rec___va_list_tag>
// CIR:   cir.va_start %[[APTR]] %[[C12345]] : !cir.ptr<!rec___va_list_tag>, !s32i
// CIR:   %[[APTR2:.+]] = cir.cast(array_to_ptrdecay, %[[ARGS]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>>), !cir.ptr<!rec___va_list_tag>
// CIR:   cir.va_end %[[APTR2]] : !cir.ptr<!rec___va_list_tag>
// CIR:   cir.return

// LLVM: define dso_local void @varargs(
// LLVM:   %[[ARGS:.+]] = alloca [1 x %struct.__va_list_tag], i64 1, align 16
// LLVM:   %[[ARGS_PTR:.+]] = getelementptr %struct.__va_list_tag, ptr %[[ARGS]], i32 0
// LLVM:   call void @llvm.va_start.p0(ptr %[[ARGS_PTR]])
// LLVM:   %[[ARGS_PTR2:.+]] = getelementptr %struct.__va_list_tag, ptr %[[ARGS]], i32 0
// LLVM:   call void @llvm.va_end.p0(ptr %[[ARGS_PTR2]])
// LLVM:   ret void

// OGCG: define dso_local void @varargs(
// OGCG:   %[[ARGS:.+]] = alloca [1 x %struct.__va_list_tag], align 16
// OGCG:   %[[ARGS_PTR:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %[[ARGS]], i64 0, i64 0
// OGCG:   call void @llvm.va_start.p0(ptr %[[ARGS_PTR]])
// OGCG:   %[[ARGS_PTR2:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %[[ARGS]], i64 0, i64 0
// OGCG:   call void @llvm.va_end.p0(ptr %[[ARGS_PTR2]])
// OGCG:   ret void

void stdarg_start(int count, ...) {
    __builtin_va_list args;
    __builtin_stdarg_start(args, 12345);
    __builtin_va_end(args);
}

// CIR-LABEL: cir.func dso_local @stdarg_start
// CIR-SAME: (%[[COUNT2:.+]]: !s32i{{.*}}, ...)
// CIR:   %[[COUNT2_ADDR:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["count", init]
// CIR:   %[[ARGS2:.+]] = cir.alloca !cir.array<!rec___va_list_tag x 1>, !cir.ptr<!cir.array<!rec___va_list_tag x 1>>, ["args"]
// CIR:   cir.store %[[COUNT2]], %[[COUNT2_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[C12345_2:.+]] = cir.const #cir.int<12345> : !s32i
// CIR:   %[[APTR3:.+]] = cir.cast(array_to_ptrdecay, %[[ARGS2]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>>), !cir.ptr<!rec___va_list_tag>
// CIR:   cir.va_start %[[APTR3]] %[[C12345_2]] : !cir.ptr<!rec___va_list_tag>, !s32i
// CIR:   %[[APTR4:.+]] = cir.cast(array_to_ptrdecay, %[[ARGS2]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>>), !cir.ptr<!rec___va_list_tag>
// CIR:   cir.va_end %[[APTR4]] : !cir.ptr<!rec___va_list_tag>
// CIR:   cir.return

// LLVM: define dso_local void @stdarg_start(
// LLVM:   %[[ARGS:.+]] = alloca [1 x %struct.__va_list_tag], i64 1, align 16
// LLVM:   %[[ARGS_PTR:.+]] = getelementptr %struct.__va_list_tag, ptr %[[ARGS]], i32 0
// LLVM:   call void @llvm.va_start.p0(ptr %[[ARGS_PTR]])
// LLVM:   %[[ARGS_PTR2:.+]] = getelementptr %struct.__va_list_tag, ptr %[[ARGS]], i32 0
// LLVM:   call void @llvm.va_end.p0(ptr %[[ARGS_PTR2]])
// LLVM:   ret void

// OGCG: define dso_local void @stdarg_start(
// OGCG:   %[[ARGS:.+]] = alloca [1 x %struct.__va_list_tag], align 16
// OGCG:   %[[ARGS_PTR:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %[[ARGS]], i64 0, i64 0
// OGCG:   call void @llvm.va_start.p0(ptr %[[ARGS_PTR]])
// OGCG:   %[[ARGS_PTR2:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %[[ARGS]], i64 0, i64 0
// OGCG:   call void @llvm.va_end.p0(ptr %[[ARGS_PTR2]])
// OGCG:   ret void
