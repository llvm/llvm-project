// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void varargs(int count, ...) {
    __builtin_va_list args;
    __builtin_va_start(args, 12345);
    __builtin_va_end(args);
}

// CIR: !rec___va_list_tag = !cir.record<struct "__va_list_tag" {!u32i, !u32i, !cir.ptr<!void>, !cir.ptr<!void>}

// CIR: cir.func dso_local @varargs(%[[COUNT_ARG:.+]]: !s32i {{.*}}, ...) {{.*}}
// CIR:   %[[COUNT:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["count", init]
// CIR:   %[[ARGS:.+]]  = cir.alloca !cir.array<!rec___va_list_tag x 1>, !cir.ptr<!cir.array<!rec___va_list_tag x 1>>, ["args"]
// CIR:   cir.store %[[COUNT_ARG]], %[[COUNT]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[ARGS_DECAY1:.+]] = cir.cast(array_to_ptrdecay, %[[ARGS]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>>), !cir.ptr<!rec___va_list_tag>
// CIR:   cir.va.start %[[ARGS_DECAY1]] : !cir.ptr<!rec___va_list_tag>
// CIR:   %[[ARGS_DECAY2:.+]] = cir.cast(array_to_ptrdecay, %[[ARGS]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>>), !cir.ptr<!rec___va_list_tag>
// CIR:   cir.va.end %[[ARGS_DECAY2]] : !cir.ptr<!rec___va_list_tag>
// CIR:   cir.return

// LLVM: %struct.__va_list_tag = type { i32, i32, ptr, ptr }

// LLVM: define dso_local void @varargs(i32 %[[ARG0:.+]], ...)
// LLVM:   %[[COUNT_ADDR:.+]] = alloca i32, i64 1
// LLVM:   %[[ARGS:.+]] = alloca [1 x %struct.__va_list_tag], i64 1
// LLVM:   store i32 %[[ARG0]], ptr %[[COUNT_ADDR]]
// LLVM:   %[[GEP1:.+]] = getelementptr %struct.__va_list_tag, ptr %[[ARGS]], i32 0
// LLVM:   call void @llvm.va_start.p0(ptr %[[GEP1]])
// LLVM:   %[[GEP2:.+]] = getelementptr %struct.__va_list_tag, ptr %[[ARGS]], i32 0
// LLVM:   call void @llvm.va_end.p0(ptr %[[GEP2]])
// LLVM:   ret void

// OGCG: %struct.__va_list_tag = type { i32, i32, ptr, ptr }

// OGCG: define dso_local void @varargs(i32 noundef %[[COUNT:.+]], ...)
// OGCG:   %[[COUNT_ADDR:.+]] = alloca i32
// OGCG:   %[[ARGS:.+]] = alloca [1 x %struct.__va_list_tag]
// OGCG:   store i32 %[[COUNT]], ptr %[[COUNT_ADDR]]
// OGCG:   %[[ARRDECAY1:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %[[ARGS]], i64 0, i64 0
// OGCG:   call void @llvm.va_start.p0(ptr %[[ARRDECAY1]])
// OGCG:   %[[ARRDECAY2:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %[[ARGS]], i64 0, i64 0
// OGCG:   call void @llvm.va_end.p0(ptr %[[ARRDECAY2]])
// OGCG:   ret void
