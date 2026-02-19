// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

struct Bar {
  float f1;
  float f2;
  unsigned u;
};

struct Bar varargs_aggregate(int count, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, count);
  struct Bar res = __builtin_va_arg(args, struct Bar);
  __builtin_va_end(args);
  return res;
}

// CIR-LABEL: cir.func {{.*}} @varargs_aggregate(
// CIR:   %[[COUNT_ADDR:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["count", init]
// CIR:   %[[RET_ADDR:.+]] = cir.alloca !rec_Bar, !cir.ptr<!rec_Bar>, ["__retval", init]
// CIR:   %[[VAAREA:.+]] = cir.alloca !cir.array<!rec___va_list_tag x 1>, !cir.ptr<!cir.array<!rec___va_list_tag x 1>>, ["args"]
// CIR:   %[[TMP_ADDR:.+]] = cir.alloca !rec_Bar, !cir.ptr<!rec_Bar>, ["vaarg.tmp"]
// CIR:   cir.store %arg0, %[[COUNT_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[VA_PTR0:.+]] = cir.cast array_to_ptrdecay %[[VAAREA]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>> -> !cir.ptr<!rec___va_list_tag>
// CIR:   %[[COUNT_VAL:.+]] = cir.load{{.*}} %[[COUNT_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.va_start %[[VA_PTR0]] %[[COUNT_VAL]] : !cir.ptr<!rec___va_list_tag>, !s32i
// CIR:   %[[VA_PTR1:.+]] = cir.cast array_to_ptrdecay %[[VAAREA]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>> -> !cir.ptr<!rec___va_list_tag>
// CIR:   %[[VA_ARG:.+]] = cir.va_arg %[[VA_PTR1]] : (!cir.ptr<!rec___va_list_tag>) -> !rec_Bar
// CIR:   cir.store{{.*}} %[[VA_ARG]], %[[TMP_ADDR]] : !rec_Bar, !cir.ptr<!rec_Bar>
// CIR:   cir.copy %[[TMP_ADDR]] to %[[RET_ADDR]] : !cir.ptr<!rec_Bar>
// CIR:   %[[VA_PTR2:.+]] = cir.cast array_to_ptrdecay %[[VAAREA]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>> -> !cir.ptr<!rec___va_list_tag>
// CIR:   cir.va_end %[[VA_PTR2]] : !cir.ptr<!rec___va_list_tag>
// CIR:   %[[RETVAL:.+]] = cir.load{{.*}} %[[RET_ADDR]] : !cir.ptr<!rec_Bar>, !rec_Bar
// CIR:   cir.return %[[RETVAL]] : !rec_Bar

// LLVM-LABEL: define dso_local %struct.Bar @varargs_aggregate(
// LLVM:   call void @llvm.va_start.p0(ptr %{{.*}})
// LLVM:   %[[VA_PTR1:.+]] = getelementptr %struct.__va_list_tag, ptr %{{.*}}, i32 0
// LLVM:   %[[VA_ARG:.+]] = va_arg ptr %[[VA_PTR1]], %struct.Bar
// LLVM:   store %struct.Bar %[[VA_ARG]], ptr %{{.*}}
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr %{{.*}}, ptr %{{.*}}, i64 12, i1 false)

// OGCG-LABEL: define dso_local { <2 x float>, i32 } @varargs_aggregate
// OGCG:   call void @llvm.va_start.p0(ptr %{{.*}})
// OGCG:   %[[VAARG_ADDR:.+]] = phi ptr [ %{{.*}}, %vaarg.in_reg ], [ %{{.*}}, %vaarg.in_mem ]
// OGCG:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 %[[VAARG_ADDR]], i64 12, i1 false)

