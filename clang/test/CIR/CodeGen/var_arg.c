// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
//
// C23 is required for __builtin_c23_va_start and for variadic functions with
// no named parameters. The LLVM checks are shared between the ClangIR
// pipeline and OG CodeGen, so they only check IR that is common to both
// pipelines.

// CIR: !rec___va_list_tag = !cir.struct<"__va_list_tag" {data !u32i, data !u32i, data !cir.ptr<!void>, data !cir.ptr<!void>}
// LLVM: %struct.__va_list_tag = type { i32, i32, ptr, ptr }

int varargs(int count, ...) {
    __builtin_va_list args;
    __builtin_va_start(args, count);
    int res = __builtin_va_arg(args, int);
    __builtin_va_end(args);
    return res;
}

// `int` classifies Extend, one INTEGER eightbyte.
// CIR-LABEL: cir.func {{.*}} @varargs(
// CIR:   %[[RET_ADDR:.+]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!s32i>
// CIR:   %[[VAAREA:.+]] = cir.alloca "args" {{.*}} : !cir.ptr<!cir.array<!rec___va_list_tag x 1>>
// CIR:   %[[RES_ADDR:.+]] = cir.alloca "res" {{.*}} init : !cir.ptr<!s32i>
// CIR:   %[[VA_PTR0:.+]] = cir.cast array_to_ptrdecay %[[VAAREA]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>> -> !cir.ptr<!rec___va_list_tag>
// CIR:   cir.va_start %[[VA_PTR0]] : !cir.ptr<!rec___va_list_tag>
// CIR:   %[[VA_PTR1:.+]] = cir.cast array_to_ptrdecay %[[VAAREA]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>> -> !cir.ptr<!rec___va_list_tag>
// CIR:   %[[GP_OFFSET_P:.+]] = cir.get_member %[[VA_PTR1]][0] {name = "gp_offset"} : !cir.ptr<!rec___va_list_tag> -> !cir.ptr<!u32i>
// CIR:   %[[GP_OFFSET:.+]] = cir.load %[[GP_OFFSET_P]] : !cir.ptr<!u32i>, !u32i
// CIR:   %[[GP_LIMIT:.+]] = cir.const #cir.int<40> : !u32i
// CIR:   %[[FITS_GP:.+]] = cir.cmp le %[[GP_OFFSET]], %[[GP_LIMIT]] : !u32i
// CIR:   %[[VA_ARG:.+]] = cir.ternary(%[[FITS_GP]], true {
// CIR:     %[[REG_SAVE:.+]] = cir.load %{{.+}}
// CIR:     %[[REG_SAVE_B:.+]] = cir.cast bitcast %[[REG_SAVE]] : !cir.ptr<!void> -> !cir.ptr<!u8i>
// CIR:     %[[REG_ADDR:.+]] = cir.ptr_stride %[[REG_SAVE_B]], %[[GP_OFFSET]] : (!cir.ptr<!u8i>, !u32i) -> !cir.ptr<!u8i>
// CIR:     %[[GP_BUMP:.+]] = cir.const #cir.int<8> : !u32i
// CIR:     %[[GP_NEXT:.+]] = cir.add %[[GP_OFFSET]], %[[GP_BUMP]] : !u32i
// CIR:     cir.store %[[GP_NEXT]], %[[GP_OFFSET_P]] : !u32i, !cir.ptr<!u32i>
// CIR:     cir.yield %[[REG_ADDR]] : !cir.ptr<!u8i>
// CIR:   }, false {
// CIR:     %[[OVERFLOW:.+]] = cir.load %{{.+}}
// CIR:     %[[OVERFLOW_B:.+]] = cir.cast bitcast %[[OVERFLOW]] : !cir.ptr<!void> -> !cir.ptr<!u8i>
// CIR:     %[[STRIDE:.+]] = cir.const #cir.int<8> : !s32i
// CIR:     %[[OVERFLOW_NEXT:.+]] = cir.ptr_stride %[[OVERFLOW_B]], %[[STRIDE]] : (!cir.ptr<!u8i>, !s32i) -> !cir.ptr<!u8i>
// CIR:     cir.yield %[[OVERFLOW_B]] : !cir.ptr<!u8i>
// CIR:   }) : (!cir.bool) -> !cir.ptr<!u8i>
// CIR:   %[[VA_ARG_B:.+]] = cir.cast bitcast %[[VA_ARG]] : !cir.ptr<!u8i> -> !cir.ptr<!s32i>
// CIR:   %[[VA_ARG_V:.+]] = cir.load %[[VA_ARG_B]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.store{{.*}} %[[VA_ARG_V]], %[[RES_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[VA_PTR2:.+]] = cir.cast array_to_ptrdecay %[[VAAREA]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>> -> !cir.ptr<!rec___va_list_tag>
// CIR:   cir.va_end %[[VA_PTR2]] : !cir.ptr<!rec___va_list_tag>
// CIR:   %[[RESULT:.+]] = cir.load{{.*}} %[[RES_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.store %[[RESULT]], %[[RET_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[RETVAL:.+]] = cir.load{{.*}} %[[RET_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.return %[[RETVAL]] : !s32i

// LLVM-LABEL: define dso_local i32 @varargs(i32 noundef %{{.+}}, ...)
// LLVM:   %[[COUNT_ADDR:.+]] = alloca i32{{.*}}, align 4
// LLVM:   %[[VAAREA:.+]] = alloca [1 x %struct.__va_list_tag]{{.*}}, align 16
// LLVM:   %[[RES_ADDR:.+]] = alloca i32{{.*}}, align 4
// LLVM:   store i32 %{{.+}}, ptr %[[COUNT_ADDR]], align 4
// LLVM:   %[[VA_PTR0:.+]] = getelementptr {{.*}}%struct.__va_list_tag{{.?}}, ptr %[[VAAREA]]
// LLVM:   call void @llvm.va_start.p0(ptr %[[VA_PTR0]])
// LLVM:   store i32 %{{.+}}, ptr %[[RES_ADDR]], align 4
// LLVM:   %[[VA_PTR1:.+]] = getelementptr {{.*}}%struct.__va_list_tag{{.?}}, ptr %[[VAAREA]]
// LLVM:   call void @llvm.va_end.p0(ptr %[[VA_PTR1]])
// LLVM:   %[[VAL:.+]] = load i32, ptr %[[RES_ADDR]], align 4
// LLVM:   ret i32 %{{.+}}

int stdarg_start(int count, ...) {
    __builtin_va_list args;
    __builtin_stdarg_start(args, 12345);
    int res = __builtin_va_arg(args, int);
    __builtin_va_end(args);
    return res;
}

// CIR-LABEL: cir.func {{.*}} @stdarg_start(
// CIR:   %[[RET_ADDR:.+]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!s32i>
// CIR:   %[[VAAREA:.+]] = cir.alloca "args" {{.*}} : !cir.ptr<!cir.array<!rec___va_list_tag x 1>>
// CIR:   %[[RES_ADDR:.+]] = cir.alloca "res" {{.*}} init : !cir.ptr<!s32i>
// CIR:   %[[VA_PTR0:.+]] = cir.cast array_to_ptrdecay %[[VAAREA]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>> -> !cir.ptr<!rec___va_list_tag>
// CIR:   cir.va_start %[[VA_PTR0]] : !cir.ptr<!rec___va_list_tag>
// CIR:   %[[VA_PTR1:.+]] = cir.cast array_to_ptrdecay %[[VAAREA]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>> -> !cir.ptr<!rec___va_list_tag>
// CIR:   %[[GP_OFFSET_P:.+]] = cir.get_member %[[VA_PTR1]][0] {name = "gp_offset"} : !cir.ptr<!rec___va_list_tag> -> !cir.ptr<!u32i>
// CIR:   %[[GP_OFFSET:.+]] = cir.load %[[GP_OFFSET_P]] : !cir.ptr<!u32i>, !u32i
// CIR:   %[[GP_LIMIT:.+]] = cir.const #cir.int<40> : !u32i
// CIR:   %[[FITS_GP:.+]] = cir.cmp le %[[GP_OFFSET]], %[[GP_LIMIT]] : !u32i
// CIR:   %[[VA_ARG:.+]] = cir.ternary(%[[FITS_GP]], true {
// CIR:     %[[REG_ADDR:.+]] = cir.ptr_stride %{{.+}}, %[[GP_OFFSET]] : (!cir.ptr<!u8i>, !u32i) -> !cir.ptr<!u8i>
// CIR:     cir.store %{{.+}}, %[[GP_OFFSET_P]] : !u32i, !cir.ptr<!u32i>
// CIR:     cir.yield %[[REG_ADDR]] : !cir.ptr<!u8i>
// CIR:   }, false {
// CIR:     cir.yield %{{.+}} : !cir.ptr<!u8i>
// CIR:   }) : (!cir.bool) -> !cir.ptr<!u8i>
// CIR:   %[[VA_ARG_B:.+]] = cir.cast bitcast %[[VA_ARG]] : !cir.ptr<!u8i> -> !cir.ptr<!s32i>
// CIR:   %[[VA_ARG_V:.+]] = cir.load %[[VA_ARG_B]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.store{{.*}} %[[VA_ARG_V]], %[[RES_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[VA_PTR2:.+]] = cir.cast array_to_ptrdecay %[[VAAREA]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>> -> !cir.ptr<!rec___va_list_tag>
// CIR:   cir.va_end %[[VA_PTR2]] : !cir.ptr<!rec___va_list_tag>
// CIR:   %[[RESULT:.+]] = cir.load{{.*}} %[[RES_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.store %[[RESULT]], %[[RET_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[RETVAL:.+]] = cir.load{{.*}} %[[RET_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.return %[[RETVAL]] : !s32i

// LLVM-LABEL: define dso_local i32 @stdarg_start(i32 noundef %{{.+}}, ...)
// LLVM:   %[[COUNT_ADDR:.+]] = alloca i32{{.*}}, align 4
// LLVM:   %[[VAAREA:.+]] = alloca [1 x %struct.__va_list_tag]{{.*}}, align 16
// LLVM:   %[[RES_ADDR:.+]] = alloca i32{{.*}}, align 4
// LLVM:   store i32 %{{.+}}, ptr %[[COUNT_ADDR]], align 4
// LLVM:   %[[VA_PTR0:.+]] = getelementptr {{.*}}%struct.__va_list_tag{{.?}}, ptr %[[VAAREA]]
// LLVM:   call void @llvm.va_start.p0(ptr %[[VA_PTR0]])
// LLVM:   store i32 %{{.+}}, ptr %[[RES_ADDR]], align 4
// LLVM:   %[[VA_PTR1:.+]] = getelementptr {{.*}}%struct.__va_list_tag{{.?}}, ptr %[[VAAREA]]
// LLVM:   call void @llvm.va_end.p0(ptr %[[VA_PTR1]])
// LLVM:   %[[VAL:.+]] = load i32, ptr %[[RES_ADDR]], align 4
// LLVM:   ret i32 %{{.+}}

void stdarg_copy() {
    __builtin_va_list src, dest;
    __builtin_va_copy(src, dest);
}

// CIR-LABEL: @stdarg_copy
// CIR:    %{{.*}} = cir.cast array_to_ptrdecay %{{.*}} : !cir.ptr<!cir.array<!rec___va_list_tag x 1>> -> !cir.ptr<!rec___va_list_tag>
// CIR:    %{{.*}} = cir.cast array_to_ptrdecay %{{.*}} : !cir.ptr<!cir.array<!rec___va_list_tag x 1>> -> !cir.ptr<!rec___va_list_tag>
// CIR:    cir.va_copy %{{.*}} to %{{.*}} : !cir.ptr<!rec___va_list_tag>, !cir.ptr<!rec___va_list_tag>

// LLVM-LABEL: define dso_local void @stdarg_copy()
// LLVM:   %[[SRC:.+]] = alloca [1 x %struct.__va_list_tag]{{.*}}, align 16
// LLVM:   %[[DEST:.+]] = alloca [1 x %struct.__va_list_tag]{{.*}}, align 16
// LLVM:   %[[SRC_PTR:.+]] = getelementptr {{.*}}%struct.__va_list_tag{{.?}}, ptr %[[SRC]]
// LLVM:   %[[DEST_PTR:.+]] = getelementptr {{.*}}%struct.__va_list_tag{{.?}}, ptr %[[DEST]]
// LLVM:   call void @llvm.va_copy.p0(ptr %[[SRC_PTR]], ptr %[[DEST_PTR]])
// LLVM:   ret void

// Test handling where the first argument is not a count, as permitted by C23.
int varargs_new(char *fmt, ...) {
    __builtin_va_list args;
    __builtin_va_start(args, fmt);
    int res = __builtin_va_arg(args, int);
    __builtin_va_end(args);
    return res;
}

// CIR-LABEL: cir.func {{.*}} @varargs_new(
// CIR:   %[[RET_ADDR:.+]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!s32i>
// CIR:   %[[VAAREA:.+]] = cir.alloca "args" {{.*}} : !cir.ptr<!cir.array<!rec___va_list_tag x 1>>
// CIR:   %[[RES_ADDR:.+]] = cir.alloca "res" {{.*}} init : !cir.ptr<!s32i>
// CIR:   %[[VA_PTR0:.+]] = cir.cast array_to_ptrdecay %[[VAAREA]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>> -> !cir.ptr<!rec___va_list_tag>
// CIR:   cir.va_start %[[VA_PTR0]] : !cir.ptr<!rec___va_list_tag>
// CIR:   %[[VA_PTR1:.+]] = cir.cast array_to_ptrdecay %[[VAAREA]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>> -> !cir.ptr<!rec___va_list_tag>
// CIR:   %[[GP_OFFSET_P:.+]] = cir.get_member %[[VA_PTR1]][0] {name = "gp_offset"} : !cir.ptr<!rec___va_list_tag> -> !cir.ptr<!u32i>
// CIR:   %[[GP_OFFSET:.+]] = cir.load %[[GP_OFFSET_P]] : !cir.ptr<!u32i>, !u32i
// CIR:   %[[GP_LIMIT:.+]] = cir.const #cir.int<40> : !u32i
// CIR:   %[[FITS_GP:.+]] = cir.cmp le %[[GP_OFFSET]], %[[GP_LIMIT]] : !u32i
// CIR:   %[[VA_ARG:.+]] = cir.ternary(%[[FITS_GP]], true {
// CIR:     %[[REG_ADDR:.+]] = cir.ptr_stride %{{.+}}, %[[GP_OFFSET]] : (!cir.ptr<!u8i>, !u32i) -> !cir.ptr<!u8i>
// CIR:     cir.store %{{.+}}, %[[GP_OFFSET_P]] : !u32i, !cir.ptr<!u32i>
// CIR:     cir.yield %[[REG_ADDR]] : !cir.ptr<!u8i>
// CIR:   }, false {
// CIR:     cir.yield %{{.+}} : !cir.ptr<!u8i>
// CIR:   }) : (!cir.bool) -> !cir.ptr<!u8i>
// CIR:   %[[VA_ARG_B:.+]] = cir.cast bitcast %[[VA_ARG]] : !cir.ptr<!u8i> -> !cir.ptr<!s32i>
// CIR:   %[[VA_ARG_V:.+]] = cir.load %[[VA_ARG_B]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.store{{.*}} %[[VA_ARG_V]], %[[RES_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[VA_PTR2:.+]] = cir.cast array_to_ptrdecay %[[VAAREA]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>> -> !cir.ptr<!rec___va_list_tag>
// CIR:   cir.va_end %[[VA_PTR2]] : !cir.ptr<!rec___va_list_tag>
// CIR:   %[[RESULT:.+]] = cir.load{{.*}} %[[RES_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.store %[[RESULT]], %[[RET_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[RETVAL:.+]] = cir.load{{.*}} %[[RET_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.return %[[RETVAL]] : !s32i

// LLVM-LABEL: define dso_local i32 @varargs_new(ptr noundef %{{.+}}, ...)
// LLVM:   %[[FMT_ADDR:.+]] = alloca ptr{{.*}}, align 8
// LLVM:   %[[VAAREA:.+]] = alloca [1 x %struct.__va_list_tag]{{.*}}, align 16
// LLVM:   %[[RES_ADDR:.+]] = alloca i32{{.*}}, align 4
// LLVM:   store ptr %{{.+}}, ptr %[[FMT_ADDR]], align 8
// LLVM:   %[[VA_PTR0:.+]] = getelementptr {{.*}}%struct.__va_list_tag{{.?}}, ptr %[[VAAREA]]
// LLVM:   call void @llvm.va_start.p0(ptr %[[VA_PTR0]])
// LLVM:   store i32 %{{.+}}, ptr %[[RES_ADDR]], align 4
// LLVM:   %[[VA_PTR1:.+]] = getelementptr {{.*}}%struct.__va_list_tag{{.?}}, ptr %[[VAAREA]]
// LLVM:   call void @llvm.va_end.p0(ptr %[[VA_PTR1]])
// LLVM:   %[[VAL:.+]] = load i32, ptr %[[RES_ADDR]], align 4
// LLVM:   ret i32 %{{.+}}

// Ensure that __builtin_va_start(list, 0) and __builtin_c23_va_start(list)
// have the same codegen.
void noargs(...) {
    __builtin_va_list list;
    __builtin_va_start(list, 0);
    __builtin_c23_va_start(list);
    __builtin_va_end(list);
}

// CIR-LABEL: cir.func {{.*}} @noargs(
// CIR:   %[[VAAREA:.+]] = cir.alloca "list" {{.*}} : !cir.ptr<!cir.array<!rec___va_list_tag x 1>>
// CIR:   %[[VA_PTR0:.+]] = cir.cast array_to_ptrdecay %[[VAAREA]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>> -> !cir.ptr<!rec___va_list_tag>
// CIR-NEXT:   cir.va_start %[[VA_PTR0]] : !cir.ptr<!rec___va_list_tag>
// CIR:   %[[VA_PTR1:.+]] = cir.cast array_to_ptrdecay %[[VAAREA]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>> -> !cir.ptr<!rec___va_list_tag>
// CIR-NEXT:   cir.va_start %[[VA_PTR1]] : !cir.ptr<!rec___va_list_tag>
// CIR:   %[[VA_PTR2:.+]] = cir.cast array_to_ptrdecay %[[VAAREA]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>> -> !cir.ptr<!rec___va_list_tag>
// CIR-NEXT:   cir.va_end %[[VA_PTR2]] : !cir.ptr<!rec___va_list_tag>

// LLVM-LABEL: define dso_local void @noargs(...)
// LLVM:   %[[VAAREA:.+]] = alloca [1 x %struct.__va_list_tag]{{.*}}, align 16
// LLVM:   %[[VA_PTR0:.+]] = getelementptr {{.*}}%struct.__va_list_tag{{.?}}, ptr %[[VAAREA]]
// LLVM:   call void @llvm.va_start.p0(ptr %[[VA_PTR0]])
// LLVM:   %[[VA_PTR1:.+]] = getelementptr {{.*}}%struct.__va_list_tag{{.?}}, ptr %[[VAAREA]]
// LLVM:   call void @llvm.va_start.p0(ptr %[[VA_PTR1]])
// LLVM:   %[[VA_PTR2:.+]] = getelementptr {{.*}}%struct.__va_list_tag{{.?}}, ptr %[[VAAREA]]
// LLVM:   call void @llvm.va_end.p0(ptr %[[VA_PTR2]])
// LLVM:   ret void

void with_param(int count, ...) {
    __builtin_va_list list;
    __builtin_c23_va_start(list, count);
    __builtin_va_end(list);
}

// CIR-LABEL: cir.func {{.*}} @with_param(
// CIR:   cir.va_start %{{.+}} : !cir.ptr<!rec___va_list_tag>
// CIR:   cir.va_end %{{.+}} : !cir.ptr<!rec___va_list_tag>

// LLVM-LABEL: define dso_local void @with_param(i32 noundef %{{.+}}, ...)
// LLVM:   %[[COUNT_ADDR:.+]] = alloca i32{{.*}}, align 4
// LLVM:   %[[VAAREA:.+]] = alloca [1 x %struct.__va_list_tag]{{.*}}, align 16
// LLVM:   store i32 %{{.+}}, ptr %[[COUNT_ADDR]], align 4
// LLVM:   %[[VA_PTR0:.+]] = getelementptr {{.*}}%struct.__va_list_tag{{.?}}, ptr %[[VAAREA]]
// LLVM:   call void @llvm.va_start.p0(ptr %[[VA_PTR0]])
// LLVM:   %[[VA_PTR1:.+]] = getelementptr {{.*}}%struct.__va_list_tag{{.?}}, ptr %[[VAAREA]]
// LLVM:   call void @llvm.va_end.p0(ptr %[[VA_PTR1]])
// LLVM:   ret void

double varargs_double(int count, ...) {
    __builtin_va_list args;
    __builtin_va_start(args, count);
    double res = __builtin_va_arg(args, double);
    __builtin_va_end(args);
    return res;
}

// `double` is Direct with no coercion, filling one vector-register eightbyte,
// so it reads the vector cursor rather than the integer one.
// CIR-LABEL: cir.func {{.*}} @varargs_double(
// CIR:   %[[FP_OFFSET_P:.+]] = cir.get_member %{{.+}}[1] {name = "fp_offset"} : !cir.ptr<!rec___va_list_tag> -> !cir.ptr<!u32i>
// CIR:   %[[FP_OFFSET:.+]] = cir.load %[[FP_OFFSET_P]] : !cir.ptr<!u32i>, !u32i
// CIR:   %[[FP_LIMIT:.+]] = cir.const #cir.int<160> : !u32i
// CIR:   cir.cmp le %[[FP_OFFSET]], %[[FP_LIMIT]] : !u32i

// LLVM-LABEL: define dso_local double @varargs_double(
// LLVM:   %[[FP_OFFSET:.+]] = load i32, ptr %{{.+}}, align {{[0-9]+}}
// LLVM:   icmp ule i32 %[[FP_OFFSET]], 160
// LLVM:   add i32 %[[FP_OFFSET]], 16
