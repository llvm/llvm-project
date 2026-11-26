// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -disable-llvm-passes -fclangir -emit-llvm %s -o %t-cir-o1.ll
// RUN: FileCheck --check-prefix=O1 --input-file=%t-cir-o1.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -disable-llvm-passes -emit-llvm %s -o %t-o1.ll
// RUN: FileCheck --check-prefix=O1 --input-file=%t-o1.ll %s

// The CIR emitted for annotations on record fields, parameters, and local
// variables, and for __builtin_annotation. The LLVM IR of the first three is
// covered by CodeGen/annotations-{field,var}.c.

struct S {
  int v __attribute__((annotate("on_field")));
};

int read_field(struct S *s) { return s->v; }

// CIR-LABEL: cir.func {{.*}} @read_field
// CIR:   %[[V:.*]] = cir.get_member %{{.*}}[0] {name = "v"} : !cir.ptr<!rec_S> -> !cir.ptr<!s32i>
// CIR:   %[[ANN:.*]] = cir.ptr_annotation %[[V]] [#cir.annotation<"on_field">] : !cir.ptr<!s32i>
// CIR:   cir.load {{.*}} %[[ANN]] : !cir.ptr<!s32i>, !s32i

int param(int p __attribute__((annotate("on_param")))) { return p; }

// CIR-LABEL: cir.func {{.*}} @param
// CIR:   %[[P:.*]] = cir.alloca "p" align(4) init : !cir.ptr<!s32i>
// CIR:   cir.var_annotation %[[P]] [#cir.annotation<"on_param">] : !cir.ptr<!s32i>

void local(void) {
  int x __attribute__((annotate("on_local", "str", 7)))
      __attribute__((annotate("second")));
  x = 1;
}

// CIR-LABEL: cir.func {{.*}} @local
// CIR:   %[[X:.*]] = cir.alloca "x" align(4) : !cir.ptr<!s32i>
// CIR:   cir.var_annotation %[[X]] [#cir.annotation<"on_local", ["str", 7 : i32]>] : !cir.ptr<!s32i>
// CIR:   cir.var_annotation %[[X]] [#cir.annotation<"second">] : !cir.ptr<!s32i>

// At -O1, the annotations are inside the variable's lifetime, as in OG.
// O1-LABEL: define {{.*}} void @local(
// O1:      %[[X:.*]] = alloca i32
// O1-NEXT: call void @llvm.lifetime.start.p0(ptr %[[X]])
// O1-NEXT: call void @llvm.var.annotation.p0.p0(ptr %[[X]],
// O1-NEXT: call void @llvm.var.annotation.p0.p0(ptr %[[X]],
// O1:      store i32 1, ptr %[[X]]
// O1:      call void @llvm.lifetime.end.p0(ptr %[[X]])
// O1:      ret void

// LLVM-DAG: [[FILE:@\.str[.0-9a-z]*]] = private unnamed_addr constant [{{[0-9]+}} x i8] c"{{.*}}annotate-local.c\00", section "llvm.metadata"
// LLVM-DAG: [[BUILTIN:@\.str[.0-9a-z]*]] = private unnamed_addr constant [11 x i8] c"on_builtin\00", section "llvm.metadata"

int builtin(int y) { return __builtin_annotation(y, "on_builtin"); }

// CIR-LABEL: cir.func {{.*}} @builtin
// CIR:   %[[Y:.*]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
// CIR:   cir.annotation %[[Y]] [#cir.annotation<"on_builtin">] : !s32i

// LLVM-LABEL: define {{.*}} i32 @builtin(
// LLVM:   call i32 @llvm.annotation.i32.p0(i32 %{{.*}}, ptr [[BUILTIN]], ptr [[FILE]], i32 [[@LINE-7]])
