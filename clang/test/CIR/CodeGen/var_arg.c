// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

int varargs(int count, ...) {
    __builtin_va_list args;
    __builtin_va_start(args, count);
    int res = __builtin_va_arg(args, int);
    __builtin_va_end(args);
    return res;
}

// CIR: !rec___va_list_tag = !cir.record<struct "__va_list_tag" {!u32i, !u32i, !cir.ptr<!void>, !cir.ptr<!void>}

// CIR: cir.func dso_local @varargs(%[[COUNT:.+]]: !s32i {{.*}}, ...) -> !s32i
// CIR:   %[[COUNT_ADDR:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["count", init]
// CIR:   %[[RETVAL:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:   %[[ARGS:.+]] = cir.alloca !cir.array<!rec___va_list_tag x 1>, !cir.ptr<!cir.array<!rec___va_list_tag x 1>>, ["args"]
// CIR:   %[[RES:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["res", init]
// CIR:   cir.store %[[COUNT]], %[[COUNT_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[ARGS_DECAY1:.+]] = cir.cast(array_to_ptrdecay, %[[ARGS]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>>), !cir.ptr<!rec___va_list_tag>
// CIR:   cir.va.start %[[ARGS_DECAY1]] : !cir.ptr<!rec___va_list_tag>
// CIR:   %[[ARGS_DECAY2:.+]] = cir.cast(array_to_ptrdecay, %[[ARGS]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>>), !cir.ptr<!rec___va_list_tag>
// CIR:   %[[ARGVAL:.+]] = cir.va.arg %[[ARGS_DECAY2]] : (!cir.ptr<!rec___va_list_tag>) -> !s32i
// CIR:   cir.store {{.*}} %[[ARGVAL]], %[[RES]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[ARGS_DECAY3:.+]] = cir.cast(array_to_ptrdecay, %[[ARGS]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>>), !cir.ptr<!rec___va_list_tag>
// CIR:   cir.va.end %[[ARGS_DECAY3]] : !cir.ptr<!rec___va_list_tag>
// CIR:   %[[RES_VAL:.+]] = cir.load {{.*}} %[[RES]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.store %[[RES_VAL]], %[[RETVAL]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[RET:.+]] = cir.load %[[RETVAL]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.return %[[RET]] : !s32i

// LLVM: %struct.__va_list_tag = type { i32, i32, ptr, ptr }

// LLVM: define dso_local i32 @varargs(i32 %[[ARG0:.+]], ...)
// LLVM:   %[[COUNT_ADDR:.+]] = alloca i32, i64 1
// LLVM:   %[[RET_SLOT:.+]] = alloca i32, i64 1
// LLVM:   %[[VASTORAGE:.+]] = alloca [1 x %struct.__va_list_tag], i64 1
// LLVM:   %[[RES:.+]] = alloca i32, i64 1
// LLVM:   store i32 %[[ARG0]], ptr %[[COUNT_ADDR]]
// LLVM:   %[[G1:.+]] = getelementptr %struct.__va_list_tag, ptr %[[VASTORAGE]], i32 0
// LLVM:   call void @llvm.va_start.p0(ptr %[[G1]])
// LLVM:   %[[G2:.+]] = getelementptr %struct.__va_list_tag, ptr %[[VASTORAGE]], i32 0
// LLVM:   %[[NEXT:.+]] = va_arg ptr %[[G2]], i32
// LLVM:   store i32 %[[NEXT]], ptr %[[RES]]
// LLVM:   %[[G3:.+]] = getelementptr %struct.__va_list_tag, ptr %[[VASTORAGE]], i32 0
// LLVM:   call void @llvm.va_end.p0(ptr %[[G3]])
// LLVM:   %[[RVAL:.+]] = load i32, ptr %[[RES]]
// LLVM:   store i32 %[[RVAL]], ptr %[[RET_SLOT]]
// LLVM:   %[[RET:.+]] = load i32, ptr %[[RET_SLOT]]
// LLVM:   ret i32 %[[RET]]

// OGCG: %struct.__va_list_tag = type { i32, i32, ptr, ptr }

// OGCG: define dso_local i32 @varargs(i32 noundef %[[COUNT:.+]], ...)
// OGCG:   %[[COUNT_ADDR:.+]] = alloca i32
// OGCG:   %[[ARGS:.+]] = alloca [1 x %struct.__va_list_tag]
// OGCG:   %[[RES:.+]] = alloca i32
// OGCG:   store i32 %[[COUNT]], ptr %[[COUNT_ADDR]]
// OGCG:   %[[DEC1:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %[[ARGS]], i64 0, i64 0
// OGCG:   call void @llvm.va_start.p0(ptr %[[DEC1]])
// OGCG:   %[[DEC2:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %[[ARGS]], i64 0, i64 0
// OGCG:   {{.*}} = getelementptr inbounds nuw %struct.__va_list_tag, ptr %[[DEC2]], i32 0, i32 0
// OGCG:   {{.*}} = load i32, ptr {{.*}}
// OGCG:   br i1 {{.*}}, label %[[INREG:.+]], label %[[INMEM:.+]]
// OGCG: [[INREG]]:
// OGCG:   {{.*}} = getelementptr inbounds nuw %struct.__va_list_tag, ptr %[[DEC2]], i32 0, i32 3
// OGCG:   {{.*}} = load ptr, ptr {{.*}}
// OGCG:   {{.*}} = getelementptr i8, ptr {{.*}}, i32 {{.*}}
// OGCG:   {{.*}} = add i32 {{.*}}, 8
// OGCG:   store i32 {{.*}}, ptr {{.*}}
// OGCG:   br label %[[END:.+]]
// OGCG: [[INMEM]]:
// OGCG:   {{.*}} = getelementptr inbounds nuw %struct.__va_list_tag, ptr %[[DEC2]], i32 0, i32 2
// OGCG:   {{.*}} = load ptr, ptr {{.*}}
// OGCG:   {{.*}} = getelementptr i8, ptr {{.*}}, i32 8
// OGCG:   store ptr {{.*}}, ptr {{.*}}
// OGCG:   br label %[[END]]
// OGCG: [[END]]:
// OGCG:   %[[ARGPTR:.+]] = phi ptr [ {{.*}}, %[[INREG]] ], [ {{.*}}, %[[INMEM]] ]
// OGCG:   %[[V:.+]] = load i32, ptr %[[ARGPTR]]
// OGCG:   store i32 %[[V]], ptr %[[RES]]
// OGCG:   %[[DEC3:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %[[ARGS]], i64 0, i64 0
// OGCG:   call void @llvm.va_end.p0(ptr %[[DEC3]])
// OGCG:   %[[RET:.+]] = load i32, ptr %[[RES]]
// OGCG:   ret i32 %[[RET]]
