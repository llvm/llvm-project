// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// Annotations on record fields, parameters, and local variables, and
// __builtin_annotation, lower to the llvm.*annotation intrinsics.

// LLVM-DAG: [[FIELD:@\.str[.0-9a-z]*]] = private unnamed_addr constant [9 x i8] c"on_field\00", section "llvm.metadata"
// LLVM-DAG: [[FILE:@\.str[.0-9a-z]*]] = private unnamed_addr constant [{{[0-9]+}} x i8] c"{{.*}}annotate-local.c\00", section "llvm.metadata"
// LLVM-DAG: [[PARAM:@\.str[.0-9a-z]*]] = private unnamed_addr constant [9 x i8] c"on_param\00", section "llvm.metadata"
// LLVM-DAG: [[LOCAL:@\.str[.0-9a-z]*]] = private unnamed_addr constant [9 x i8] c"on_local\00", section "llvm.metadata"
// LLVM-DAG: [[ARGSTR:@\.str[.0-9a-z]*]] = private unnamed_addr constant [4 x i8] c"str\00", align 1
// LLVM-DAG: [[ARGS:@\.args[.0-9a-z]*]] = private unnamed_addr constant { ptr, i32 } { ptr [[ARGSTR]], i32 7 }, section "llvm.metadata"
// LLVM-DAG: [[SECOND:@\.str[.0-9a-z]*]] = private unnamed_addr constant [7 x i8] c"second\00", section "llvm.metadata"
// LLVM-DAG: [[BUILTIN:@\.str[.0-9a-z]*]] = private unnamed_addr constant [11 x i8] c"on_builtin\00", section "llvm.metadata"

struct S {
  int v __attribute__((annotate("on_field")));
};

int read_field(struct S *s) { return s->v; }

// CIR-LABEL: cir.func {{.*}} @read_field
// CIR:   %[[V:.*]] = cir.get_member %{{.*}}[0] {name = "v"} : !cir.ptr<!rec_S> -> !cir.ptr<!s32i>
// CIR:   %[[ANN:.*]] = cir.ptr_annotation %[[V]] [#cir.annotation<"on_field">] : !cir.ptr<!s32i>
// CIR:   cir.load {{.*}} %[[ANN]] : !cir.ptr<!s32i>, !s32i

// LLVM-LABEL: define {{.*}} i32 @read_field(
// LLVM:   %[[ANN:.*]] = call ptr @llvm.ptr.annotation.p0.p0(ptr %{{.*}}, ptr [[FIELD]], ptr [[FILE]], i32 [[@LINE-11]], ptr null)
// LLVM:   load i32, ptr %[[ANN]]

int param(int p __attribute__((annotate("on_param")))) { return p; }

// CIR-LABEL: cir.func {{.*}} @param
// CIR:   %[[P:.*]] = cir.alloca "p" align(4) init : !cir.ptr<!s32i>
// CIR:   cir.var_annotation %[[P]] [#cir.annotation<"on_param">] : !cir.ptr<!s32i>

// LLVM-LABEL: define {{.*}} i32 @param(
// LLVM:   call void @llvm.var.annotation.p0.p0(ptr %{{.*}}, ptr [[PARAM]], ptr [[FILE]], i32 [[@LINE-7]], ptr null)

void local(void) {
  int x __attribute__((annotate("on_local", "str", 7)))
      __attribute__((annotate("second")));
  x = 1;
}

// CIR-LABEL: cir.func {{.*}} @local
// CIR:   %[[X:.*]] = cir.alloca "x" align(4) : !cir.ptr<!s32i>
// CIR:   cir.var_annotation %[[X]] [#cir.annotation<"on_local", ["str", 7 : i32]>] : !cir.ptr<!s32i>
// CIR:   cir.var_annotation %[[X]] [#cir.annotation<"second">] : !cir.ptr<!s32i>

// LLVM-LABEL: define {{.*}} void @local(
// LLVM:   %[[X:.*]] = alloca i32
// LLVM:   call void @llvm.var.annotation.p0.p0(ptr %[[X]], ptr [[LOCAL]], ptr [[FILE]], i32 [[@LINE-12]], ptr [[ARGS]])
// LLVM:   call void @llvm.var.annotation.p0.p0(ptr %[[X]], ptr [[SECOND]], ptr [[FILE]], i32 [[@LINE-13]], ptr null)

int builtin(int y) { return __builtin_annotation(y, "on_builtin"); }

// CIR-LABEL: cir.func {{.*}} @builtin
// CIR:   %[[Y:.*]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
// CIR:   cir.annotation %[[Y]] [#cir.annotation<"on_builtin">] : !s32i

// LLVM-LABEL: define {{.*}} i32 @builtin(
// LLVM:   call i32 @llvm.annotation.i32.p0(i32 %{{.*}}, ptr [[BUILTIN]], ptr [[FILE]], i32 [[@LINE-7]])
