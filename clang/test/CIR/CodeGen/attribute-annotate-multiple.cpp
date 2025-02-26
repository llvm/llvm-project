// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=BEFORE
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir -mmlir --mlir-print-ir-after=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=AFTER
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
double *a __attribute__((annotate("withargs", "21", 12 )));
int *b __attribute__((annotate("withargs", "21", 12 )));
void *c __attribute__((annotate("noargvar")));

enum : char { npu1 = 42};
int tile __attribute__((annotate("cir.aie.device.tile", npu1))) = 7;

void foo(int i) __attribute__((annotate("noargfunc"))) {
}
// redeclare with more annotate
void foo(int i) __attribute__((annotate("withargfunc", "os", 23 )));
void bar() __attribute__((annotate("withargfunc", "os", 22))) {
}

// BEFORE: module @{{.*}}attribute-annotate-multiple.cpp" attributes {{{.*}}cir.lang =

// BEFORE: cir.global external @a = #cir.ptr<null> : !cir.ptr<!cir.double>
// BEFORE-SAME: [#cir.annotation<name = "withargs", args = ["21", 12 : i32]>]
// BEFORE: cir.global external @b = #cir.ptr<null> : !cir.ptr<!s32i>
// BEFORE-SAME: [#cir.annotation<name = "withargs", args = ["21", 12 : i32]>]
// BEFORE: cir.global external @c = #cir.ptr<null> : !cir.ptr<!void>
// BEFORE-SAME: [#cir.annotation<name = "noargvar", args = []>]
// BEFORE: cir.global external @tile = #cir.int<7> : !s32i
// BEFORE-SAME: #cir.annotation<name = "cir.aie.device.tile", args = [42 : i8]>]

// BEFORE: cir.func @_Z3fooi(%arg0: !s32i) [#cir.annotation<name = "noargfunc", args = []>,
// BEFORE-SAME: #cir.annotation<name = "withargfunc", args = ["os", 23 : i32]>]
// BEFORE: cir.func @_Z3barv() [#cir.annotation<name = "withargfunc", args = ["os", 22 : i32]>]


// AFTER: module {{.*}}attribute-annotate-multiple.cpp" attributes
// AFTER-SAME: {cir.global_annotations = #cir<global_annotations [
// AFTER-SAME: ["a", #cir.annotation<name = "withargs", args = ["21", 12 : i32]>],
// AFTER-SAME: ["b", #cir.annotation<name = "withargs", args = ["21", 12 : i32]>],
// AFTER-SAME: ["c", #cir.annotation<name = "noargvar", args = []>],
// AFTER-SAME: ["tile", #cir.annotation<name = "cir.aie.device.tile", args = [42 : i8]>],
// AFTER-SAME: ["_Z3fooi", #cir.annotation<name = "noargfunc", args = []>],
// AFTER-SAME: ["_Z3fooi", #cir.annotation<name = "withargfunc", args = ["os", 23 : i32]>],
// AFTER-SAME: ["_Z3barv", #cir.annotation<name = "withargfunc", args = ["os", 22 : i32]>]]>,


// LLVM: @a = global ptr null
// LLVM: @b = global ptr null
// LLVM: @c = global ptr null
// LLVM: @tile = global i32 7
// LLVM: @.str.annotation = private unnamed_addr constant [9 x i8] c"withargs\00", section "llvm.metadata"
// LLVM: @.str.1.annotation = private unnamed_addr constant [{{[0-9]+}} x i8] c"{{.*}}attribute-annotate-multiple.cpp\00", section "llvm.metadata"
// LLVM: @.str.annotation.arg = private unnamed_addr constant [3 x i8] c"21\00", align 1
// LLVM: @.args.annotation = private unnamed_addr constant { ptr, i32 } { ptr @.str.annotation.arg, i32 12 }, section "llvm.metadata"
// LLVM: @.str.2.annotation = private unnamed_addr constant [9 x i8] c"noargvar\00", section "llvm.metadata"
// LLVM: @.str.3.annotation = private unnamed_addr constant [20 x i8] c"cir.aie.device.tile\00", section "llvm.metadata"
// LLVM: @.args.1.annotation = private unnamed_addr constant { i8 } { i8 42 }, section "llvm.metadata"
// LLVM: @.str.4.annotation = private unnamed_addr constant [10 x i8] c"noargfunc\00", section "llvm.metadata"
// LLVM: @.str.5.annotation = private unnamed_addr constant [12 x i8] c"withargfunc\00", section "llvm.metadata"
// LLVM: @.str.1.annotation.arg = private unnamed_addr constant [3 x i8] c"os\00", align 1
// LLVM: @.args.2.annotation = private unnamed_addr constant { ptr, i32 } 
// LLVM-SAME: { ptr @.str.1.annotation.arg, i32 23 }, section "llvm.metadata"
// LLVM: @.args.3.annotation = private unnamed_addr constant { ptr, i32 } 
// LLVM-SAME: { ptr @.str.1.annotation.arg, i32 22 }, section "llvm.metadata"

// LLVM: @llvm.global.annotations = appending global [7 x { ptr, ptr, ptr, i32, ptr }]
// LLVM-SAME: [{ ptr, ptr, ptr, i32, ptr }
// LLVM-SAME: { ptr @a, ptr @.str.annotation, ptr @.str.1.annotation, i32 5, ptr @.args.annotation },
// LLVM-SAME: { ptr, ptr, ptr, i32, ptr }
// LLVM-SAME: { ptr @b, ptr @.str.annotation, ptr @.str.1.annotation, i32 6, ptr @.args.annotation },
// LLVM-SAME: { ptr, ptr, ptr, i32, ptr }
// LLVM-SAME: { ptr @c, ptr @.str.2.annotation, ptr @.str.1.annotation, i32 7, ptr null },
// LLVM-SAME: { ptr, ptr, ptr, i32, ptr }
// LLVM-SAME: { ptr @tile, ptr @.str.3.annotation, ptr @.str.1.annotation, i32 10, ptr @.args.1.annotation },
// LLVM-SAME: { ptr, ptr, ptr, i32, ptr }
// LLVM-SAME: { ptr @_Z3fooi, ptr @.str.4.annotation, ptr @.str.1.annotation, i32 12, ptr null },
// LLVM-SAME: { ptr, ptr, ptr, i32, ptr }
// LLVM-SAME: { ptr @_Z3fooi, ptr @.str.5.annotation, ptr @.str.1.annotation, i32 12, ptr @.args.2.annotation },
// LLVM-SAME: { ptr, ptr, ptr, i32, ptr }
// LLVM-SAME: { ptr @_Z3barv, ptr @.str.5.annotation, ptr @.str.1.annotation, i32 16, ptr @.args.3.annotation }],
// LLVM-SAME: section "llvm.metadata"

// LLVM: define dso_local void @_Z3fooi(i32 %0)
// LLVM: define dso_local void @_Z3barv()
