// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefixes=LLVM,LLVMCIR
// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=LLVM,OGCG
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

// `int` is INTEGER class, one eightbyte, so the fetch gates on gp_offset.
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
// CIR:   %[[VA_ARG_V:.+]] = cir.load align(4) %[[VA_ARG_B]] : !cir.ptr<!s32i>, !s32i
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
// CIR:   %[[VA_ARG_V:.+]] = cir.load align(4) %[[VA_ARG_B]] : !cir.ptr<!s32i>, !s32i
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
// CIR:   %[[VA_ARG_V:.+]] = cir.load align(4) %[[VA_ARG_B]] : !cir.ptr<!s32i>, !s32i
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
// CIR:   %[[FITS:.+]] = cir.cmp le %[[FP_OFFSET]], %[[FP_LIMIT]] : !u32i
// CIR:   %[[ADDR:.+]] = cir.ternary(%[[FITS]], true {
// CIR:     %[[REG_SAVE_B:.+]] = cir.cast bitcast %{{.+}} : !cir.ptr<!void> -> !cir.ptr<!u8i>
// CIR:     %[[REG_ADDR:.+]] = cir.ptr_stride %[[REG_SAVE_B]], %[[FP_OFFSET]] : (!cir.ptr<!u8i>, !u32i) -> !cir.ptr<!u8i>
// Taking the register also spends it, so the cursor moves on by one slot.
// CIR:     %[[STEP:.+]] = cir.const #cir.int<16> : !u32i
// CIR:     %[[FP_NEXT:.+]] = cir.add %[[FP_OFFSET]], %[[STEP]] : !u32i
// CIR:     cir.store %[[FP_NEXT]], %[[FP_OFFSET_P]] : !u32i, !cir.ptr<!u32i>
// CIR:     cir.yield %[[REG_ADDR]] : !cir.ptr<!u8i>
// CIR:   }, false {
// The overflow arm reads at the cursor and advances it by the argument size.
// CIR:     %[[OVERFLOW_P:.+]] = cir.get_member %{{.+}}[2] {name = "overflow_arg_area"} : !cir.ptr<!rec___va_list_tag> -> !cir.ptr<!cir.ptr<!void>>
// CIR:     %[[OVERFLOW:.+]] = cir.load %[[OVERFLOW_P]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR:     %[[OVERFLOW_B:.+]] = cir.cast bitcast %[[OVERFLOW]] : !cir.ptr<!void> -> !cir.ptr<!u8i>
// CIR:     %[[STRIDE:.+]] = cir.const #cir.int<8> : !s32i
// CIR:     %[[MEM_NEXT:.+]] = cir.ptr_stride %[[OVERFLOW_B]], %[[STRIDE]] : (!cir.ptr<!u8i>, !s32i) -> !cir.ptr<!u8i>
// CIR:     cir.store %[[MEM_NEXT]], %{{.+}} : !cir.ptr<!u8i>, !cir.ptr<!cir.ptr<!u8i>>
// CIR:     cir.yield %[[OVERFLOW_B]] : !cir.ptr<!u8i>
// CIR:   }) : (!cir.bool) -> !cir.ptr<!u8i>
// CIR:   %[[RESULT_P:.+]] = cir.cast bitcast %[[ADDR]] : !cir.ptr<!u8i> -> !cir.ptr<!cir.double>
// CIR:   cir.load align(8) %[[RESULT_P]] : !cir.ptr<!cir.double>, !cir.double

// LLVM-LABEL: define dso_local double @varargs_double(
// LLVM:   %[[FP_OFFSET_P:.+]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr %{{.+}}, i32 0, i32 1
// LLVM:   %[[FP_OFFSET:.+]] = load i32, ptr %[[FP_OFFSET_P]], align 4
// LLVM:   icmp ule i32 %[[FP_OFFSET]], 160
// LLVMCIR: %[[RSA:.+]] = load ptr, ptr %{{.+}}, align 8
// OGCG:    %[[RSA:.+]] = load ptr, ptr %{{.+}}, align 16
// LLVMCIR: %[[FP64:.+]] = zext i32 %[[FP_OFFSET]] to i64
// LLVMCIR: %[[REG_ADDR:.+]] = getelementptr i8, ptr %[[RSA]], i64 %[[FP64]]
// OGCG:    %[[REG_ADDR:.+]] = getelementptr i8, ptr %[[RSA]], i32 %[[FP_OFFSET]]
// LLVM:   %[[FP_NEXT:.+]] = add i32 %[[FP_OFFSET]], 16
// LLVM:   store i32 %[[FP_NEXT]], ptr %[[FP_OFFSET_P]], align 4
// LLVM:   %[[OVERFLOW_P:.+]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr %{{.+}}, i32 0, i32 2
// LLVM:   %[[OVERFLOW:.+]] = load ptr, ptr %[[OVERFLOW_P]], align 8
// LLVM:   %[[MEM_NEXT:.+]] = getelementptr i8, ptr %[[OVERFLOW]], i{{32|64}} 8
// LLVM:   store ptr %[[MEM_NEXT]], ptr %[[OVERFLOW_P]], align 8
// LLVMCIR: %[[ADDR:.+]] = phi ptr [ %[[OVERFLOW]], %{{.+}} ], [ %[[REG_ADDR]], %{{.+}} ]
// OGCG:    %[[ADDR:.+]] = phi ptr [ %[[REG_ADDR]], %{{.+}} ], [ %[[OVERFLOW]], %{{.+}} ]
// LLVM:   load double, ptr %[[ADDR]], align 8

short varargs_short(int count, ...) {
    __builtin_va_list args;
    __builtin_va_start(args, count);
    short res = __builtin_va_arg(args, short);
    __builtin_va_end(args);
    return res;
}

// A type narrower than a register is sign-extended into one.  That is a
// separate classification from the `int` above, though it produces the same
// one-register sequence.
// CIR-LABEL: cir.func {{.*}} @varargs_short(
// CIR:   %[[GP_OFFSET_P:.+]] = cir.get_member %{{.+}}[0] {name = "gp_offset"} : !cir.ptr<!rec___va_list_tag> -> !cir.ptr<!u32i>
// CIR:   %[[GP_OFFSET:.+]] = cir.load %[[GP_OFFSET_P]] : !cir.ptr<!u32i>, !u32i
// CIR:   %[[GP_LIMIT:.+]] = cir.const #cir.int<40> : !u32i
// CIR:   %[[FITS:.+]] = cir.cmp le %[[GP_OFFSET]], %[[GP_LIMIT]] : !u32i
// CIR:   %[[ADDR:.+]] = cir.ternary(%[[FITS]], true {
// CIR:     %[[REG_SAVE_B:.+]] = cir.cast bitcast %{{.+}} : !cir.ptr<!void> -> !cir.ptr<!u8i>
// CIR:     %[[REG_ADDR:.+]] = cir.ptr_stride %[[REG_SAVE_B]], %[[GP_OFFSET]] : (!cir.ptr<!u8i>, !u32i) -> !cir.ptr<!u8i>
// CIR:     %[[STEP:.+]] = cir.const #cir.int<8> : !u32i
// CIR:     %[[GP_NEXT:.+]] = cir.add %[[GP_OFFSET]], %[[STEP]] : !u32i
// CIR:     cir.store %[[GP_NEXT]], %[[GP_OFFSET_P]] : !u32i, !cir.ptr<!u32i>
// CIR:     cir.yield %[[REG_ADDR]] : !cir.ptr<!u8i>
// CIR:   }) : (!cir.bool) -> !cir.ptr<!u8i>
// CIR:   %[[RESULT_P:.+]] = cir.cast bitcast %[[ADDR]] : !cir.ptr<!u8i> -> !cir.ptr<!s16i>
// CIR:   cir.load align(2) %[[RESULT_P]] : !cir.ptr<!s16i>, !s16i

// LLVM-LABEL: define dso_local signext i16 @varargs_short(i32 noundef %{{.*}}, ...)
// LLVM:   %[[GP_OFFSET_P:.+]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr %{{.+}}, i32 0, i32 0
// LLVMCIR: %[[GP_OFFSET:.+]] = load i32, ptr %[[GP_OFFSET_P]], align 4
// OGCG:    %[[GP_OFFSET:.+]] = load i32, ptr %[[GP_OFFSET_P]], align 16
// LLVM:   icmp ule i32 %[[GP_OFFSET]], 40
// LLVMCIR: %[[RSA:.+]] = load ptr, ptr %{{.+}}, align 8
// OGCG:    %[[RSA:.+]] = load ptr, ptr %{{.+}}, align 16
// LLVMCIR: %[[GP64:.+]] = zext i32 %[[GP_OFFSET]] to i64
// LLVMCIR: %[[REG_ADDR:.+]] = getelementptr i8, ptr %[[RSA]], i64 %[[GP64]]
// OGCG:    %[[REG_ADDR:.+]] = getelementptr i8, ptr %[[RSA]], i32 %[[GP_OFFSET]]
// LLVM:   %[[GP_NEXT:.+]] = add i32 %[[GP_OFFSET]], 8
// LLVMCIR: store i32 %[[GP_NEXT]], ptr %[[GP_OFFSET_P]], align 4
// OGCG:    store i32 %[[GP_NEXT]], ptr %[[GP_OFFSET_P]], align 16
// LLVM:   %[[OVERFLOW:.+]] = load ptr, ptr %[[OVERFLOW_P:.+]], align 8
// LLVM:   %[[MEM_NEXT:.+]] = getelementptr i8, ptr %[[OVERFLOW]], i{{32|64}} 8
// LLVM:   store ptr %[[MEM_NEXT]], ptr %[[OVERFLOW_P]], align 8
// LLVMCIR: %[[ADDR:.+]] = phi ptr [ %[[OVERFLOW]], %{{.+}} ], [ %[[REG_ADDR]], %{{.+}} ]
// OGCG:    %[[ADDR:.+]] = phi ptr [ %[[REG_ADDR]], %{{.+}} ], [ %[[OVERFLOW]], %{{.+}} ]
// LLVM:   load i16, ptr %[[ADDR]], align 2
