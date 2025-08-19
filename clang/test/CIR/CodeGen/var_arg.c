// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// CIR: !rec___va_list_tag = !cir.record<struct "__va_list_tag" {!u32i, !u32i, !cir.ptr<!void>, !cir.ptr<!void>}
// LLVM: %struct.__va_list_tag = type { i32, i32, ptr, ptr }
// OGCG: %struct.__va_list_tag = type { i32, i32, ptr, ptr }

int varargs(int count, ...) {
    __builtin_va_list args;
    __builtin_va_start(args, count);
    int res = __builtin_va_arg(args, int);
    __builtin_va_end(args);
    return res;
}

// CIR-LABEL: cir.func dso_local @varargs(
// CIR:   %[[COUNT_ADDR:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["count", init]
// CIR:   %[[RET_ADDR:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:   %[[VAAREA:.+]] = cir.alloca !cir.array<!rec___va_list_tag x 1>, !cir.ptr<!cir.array<!rec___va_list_tag x 1>>, ["args"]
// CIR:   %[[RES_ADDR:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["res", init]
// CIR:   cir.store %arg0, %[[COUNT_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[VA_PTR0:.+]] = cir.cast(array_to_ptrdecay, %[[VAAREA]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>>), !cir.ptr<!rec___va_list_tag>
// CIR:   %[[COUNT_VAL:.+]] = cir.load{{.*}} %[[COUNT_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.va_start %[[VA_PTR0]] %[[COUNT_VAL]] : !cir.ptr<!rec___va_list_tag>, !s32i
// CIR:   %[[VA_PTR1:.+]] = cir.cast(array_to_ptrdecay, %[[VAAREA]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>>), !cir.ptr<!rec___va_list_tag>
// CIR:   %[[VA_ARG:.+]] = cir.va_arg %[[VA_PTR1]] : (!cir.ptr<!rec___va_list_tag>) -> !s32i
// CIR:   cir.store{{.*}} %[[VA_ARG]], %[[RES_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[VA_PTR2:.+]] = cir.cast(array_to_ptrdecay, %[[VAAREA]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>>), !cir.ptr<!rec___va_list_tag>
// CIR:   cir.va_end %[[VA_PTR2]] : !cir.ptr<!rec___va_list_tag>
// CIR:   %[[RESULT:.+]] = cir.load{{.*}} %[[RES_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.store %[[RESULT]], %[[RET_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[RETVAL:.+]] = cir.load{{.*}} %[[RET_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.return %[[RETVAL]] : !s32i

// LLVM-LABEL: define dso_local i32 @varargs(
// LLVM:   %[[COUNT_ADDR:.+]] = alloca i32{{.*}}
// LLVM:   %[[RET_ADDR:.+]] = alloca i32{{.*}}
// LLVM:   %[[VAAREA:.+]] = alloca [1 x %struct.__va_list_tag]{{.*}}
// LLVM:   %[[RES_ADDR:.+]] = alloca i32{{.*}}
// LLVM:   %[[VA_PTR0:.+]] = getelementptr %struct.__va_list_tag, ptr %[[VAAREA]], i32 0
// LLVM:   call void @llvm.va_start.p0(ptr %[[VA_PTR0]])
// LLVM:   %[[VA_PTR1:.+]] = getelementptr %struct.__va_list_tag, ptr %[[VAAREA]], i32 0
// LLVM:   %[[VA_ARG:.+]] = va_arg ptr %[[VA_PTR1]], i32
// LLVM:   store i32 %[[VA_ARG]], ptr %[[RES_ADDR]], {{.*}}
// LLVM:   %[[VA_PTR2:.+]] = getelementptr %struct.__va_list_tag, ptr %[[VAAREA]], i32 0
// LLVM:   call void @llvm.va_end.p0(ptr %[[VA_PTR2]])
// LLVM:   %[[TMP_LOAD:.+]] = load i32, ptr %[[RES_ADDR]], {{.*}}
// LLVM:   store i32 %[[TMP_LOAD]], ptr %[[RET_ADDR]], {{.*}}
// LLVM:   %[[RETVAL:.+]] = load i32, ptr %[[RET_ADDR]], {{.*}}
// LLVM:   ret i32 %[[RETVAL]]

// OGCG-LABEL: define dso_local i32 @varargs
// OGCG:   %[[COUNT_ADDR:.+]] = alloca i32
// OGCG:   %[[VAAREA:.+]] = alloca [1 x %struct.__va_list_tag]
// OGCG:   %[[RES_ADDR:.+]] = alloca i32
// OGCG:   %[[DECAY:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %[[VAAREA]]
// OGCG:   call void @llvm.va_start.p0(ptr %[[DECAY]])
// OGCG:   %[[DECAY1:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %[[VAAREA]]
// OGCG:   %[[GPOFFSET_PTR:.+]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr %[[DECAY1]], i32 0, i32 0
// OGCG:   %[[GPOFFSET:.+]] = load i32, ptr %[[GPOFFSET_PTR]]
// OGCG:   %[[COND:.+]] = icmp ule i32 %[[GPOFFSET]], 40
// OGCG:   br i1 %[[COND]], label %vaarg.in_reg, label %vaarg.in_mem
//
// OGCG: vaarg.in_reg:
// OGCG:   %[[REGSAVE_PTR:.+]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr %[[DECAY1]], i32 0, i32 3
// OGCG:   %[[REGSAVE:.+]] = load ptr, ptr %[[REGSAVE_PTR]]
// OGCG:   %[[VAADDR1:.+]] = getelementptr i8, ptr %[[REGSAVE]], i32 %[[GPOFFSET]]
// OGCG:   br label %vaarg.end
//
// OGCG: vaarg.in_mem:
// OGCG:   %[[OVERFLOW_PTR:.+]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr %[[DECAY1]], i32 0, i32 2
// OGCG:   %[[OVERFLOW:.+]] = load ptr, ptr %[[OVERFLOW_PTR]]
// OGCG:   br label %vaarg.end
//
// OGCG: vaarg.end:
// OGCG:   %[[PHI:.+]] = phi ptr [ %[[VAADDR1]], %vaarg.in_reg ], [ %[[OVERFLOW]], %vaarg.in_mem ]
// OGCG:   %[[LOADED:.+]] = load i32, ptr %[[PHI]]
// OGCG:   store i32 %[[LOADED]], ptr %[[RES_ADDR]]
// OGCG:   %[[DECAY2:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %[[VAAREA]]
// OGCG:   call void @llvm.va_end.p0(ptr %[[DECAY2]])
// OGCG:   %[[VAL:.+]] = load i32, ptr %[[RES_ADDR]]
// OGCG:   ret i32 %[[VAL]]

int stdarg_start(int count, ...) {
    __builtin_va_list args;
    __builtin_stdarg_start(args, 12345);
    int res = __builtin_va_arg(args, int);
    __builtin_va_end(args);
    return res;
}

// CIR-LABEL: cir.func dso_local @stdarg_start(
// CIR:   %[[COUNT_ADDR:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["count", init]
// CIR:   %[[RET_ADDR:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:   %[[VAAREA:.+]] = cir.alloca !cir.array<!rec___va_list_tag x 1>, !cir.ptr<!cir.array<!rec___va_list_tag x 1>>, ["args"]
// CIR:   %[[RES_ADDR:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["res", init]
// CIR:   cir.store %arg0, %[[COUNT_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[VA_PTR0:.+]] = cir.cast(array_to_ptrdecay, %[[VAAREA]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>>), !cir.ptr<!rec___va_list_tag>
// CIR:   %[[C12345:.+]] = cir.const #cir.int<12345> : !s32i
// CIR:   cir.va_start %[[VA_PTR0]] %[[C12345]] : !cir.ptr<!rec___va_list_tag>, !s32i
// CIR:   %[[VA_PTR1:.+]] = cir.cast(array_to_ptrdecay, %[[VAAREA]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>>), !cir.ptr<!rec___va_list_tag>
// CIR:   %[[VA_ARG:.+]] = cir.va_arg %[[VA_PTR1]] : (!cir.ptr<!rec___va_list_tag>) -> !s32i
// CIR:   cir.store{{.*}} %[[VA_ARG]], %[[RES_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[VA_PTR2:.+]] = cir.cast(array_to_ptrdecay, %[[VAAREA]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>>), !cir.ptr<!rec___va_list_tag>
// CIR:   cir.va_end %[[VA_PTR2]] : !cir.ptr<!rec___va_list_tag>
// CIR:   %[[RESULT:.+]] = cir.load{{.*}} %[[RES_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.store %[[RESULT]], %[[RET_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[RETVAL:.+]] = cir.load{{.*}} %[[RET_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.return %[[RETVAL]] : !s32i

// LLVM-LABEL: define dso_local i32 @stdarg_start(
// LLVM:   %[[COUNT_ADDR:.+]] = alloca i32{{.*}}
// LLVM:   %[[RET_ADDR:.+]] = alloca i32{{.*}}
// LLVM:   %[[VAAREA:.+]] = alloca [1 x %struct.__va_list_tag]{{.*}}
// LLVM:   %[[RES_ADDR:.+]] = alloca i32{{.*}}
// LLVM:   %[[VA_PTR0:.+]] = getelementptr %struct.__va_list_tag, ptr %[[VAAREA]], i32 0
// LLVM:   call void @llvm.va_start.p0(ptr %[[VA_PTR0]])
// LLVM:   %[[VA_PTR1:.+]] = getelementptr %struct.__va_list_tag, ptr %[[VAAREA]], i32 0
// LLVM:   %[[VA_ARG:.+]] = va_arg ptr %[[VA_PTR1]], i32
// LLVM:   store i32 %[[VA_ARG]], ptr %[[RES_ADDR]], {{.*}}
// LLVM:   %[[VA_PTR2:.+]] = getelementptr %struct.__va_list_tag, ptr %[[VAAREA]], i32 0
// LLVM:   call void @llvm.va_end.p0(ptr %[[VA_PTR2]])
// LLVM:   %[[TMP_LOAD:.+]] = load i32, ptr %[[RES_ADDR]], {{.*}}
// LLVM:   store i32 %[[TMP_LOAD]], ptr %[[RET_ADDR]], {{.*}}
// LLVM:   %[[RETVAL:.+]] = load i32, ptr %[[RET_ADDR]], {{.*}}
// LLVM:   ret i32 %[[RETVAL]]

// OGCG-LABEL: define dso_local i32 @stdarg_start
// OGCG:   %[[COUNT_ADDR:.+]] = alloca i32
// OGCG:   %[[VAAREA:.+]] = alloca [1 x %struct.__va_list_tag]
// OGCG:   %[[RES_ADDR:.+]] = alloca i32
// OGCG:   %[[DECAY:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %[[VAAREA]], i64 0, i64 0
// OGCG:   call void @llvm.va_start.p0(ptr %[[DECAY]])
// OGCG:   %[[DECAY1:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %[[VAAREA]], i64 0, i64 0
// OGCG:   %[[GPOFFSET_PTR:.+]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr %[[DECAY1]], i32 0, i32 0
// OGCG:   %[[GPOFFSET:.+]] = load i32, ptr %[[GPOFFSET_PTR]]
// OGCG:   %[[COND:.+]] = icmp ule i32 %[[GPOFFSET]], 40
// OGCG:   br i1 %[[COND]], label %vaarg.in_reg, label %vaarg.in_mem
//
// OGCG: vaarg.in_reg:                                   
// OGCG:   %[[REGSAVE_PTR:.+]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr %[[DECAY1]], i32 0, i32 3
// OGCG:   %[[REGSAVE:.+]] = load ptr, ptr %[[REGSAVE_PTR]]
// OGCG:   %[[VAADDR1:.+]] = getelementptr i8, ptr %[[REGSAVE]], i32 %[[GPOFFSET]]
// OGCG:   %[[NEXT_GPOFFSET:.+]] = add i32 %[[GPOFFSET]], 8
// OGCG:   store i32 %[[NEXT_GPOFFSET]], ptr %[[GPOFFSET_PTR]]
// OGCG:   br label %vaarg.end
//
// OGCG: vaarg.in_mem:
// OGCG:   %[[OVERFLOW_PTR:.+]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr %[[DECAY1]], i32 0, i32 2
// OGCG:   %[[OVERFLOW:.+]] = load ptr, ptr %[[OVERFLOW_PTR]]
// OGCG:   %[[OVERFLOW_NEXT:.+]] = getelementptr i8, ptr %[[OVERFLOW]], i32 8
// OGCG:   store ptr %[[OVERFLOW_NEXT]], ptr %[[OVERFLOW_PTR]]
// OGCG:   br label %vaarg.end
//
// OGCG: vaarg.end:
// OGCG:   %[[PHI:.+]] = phi ptr [ %[[VAADDR1]], %vaarg.in_reg ], [ %[[OVERFLOW]], %vaarg.in_mem ]
// OGCG:   %[[LOADED:.+]] = load i32, ptr %[[PHI]]
// OGCG:   store i32 %[[LOADED]], ptr %[[RES_ADDR]]
// OGCG:   %[[DECAY2:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %[[VAAREA]], i64 0, i64 0
// OGCG:   call void @llvm.va_end.p0(ptr %[[DECAY2]])
// OGCG:   %[[VAL:.+]] = load i32, ptr %[[RES_ADDR]]
// OGCG:   ret i32 %[[VAL]]
