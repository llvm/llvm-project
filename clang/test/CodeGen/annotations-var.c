// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o %t1 %s
// RUN: FileCheck --check-prefix=LOCAL %s < %t1
// RUN: FileCheck --check-prefix=UNDEF %s < %t1
// RUN: FileCheck --check-prefix=PARAM %s < %t1
// END.

// LOCAL: private unnamed_addr constant [15 x i8] c"localvar_ann_{{.}}\00", section "llvm.metadata"
// LOCAL: private unnamed_addr constant [15 x i8] c"localvar_ann_{{.}}\00", section "llvm.metadata"

// UNDEF: private unnamed_addr constant [15 x i8] c"undefvar_ann_0\00", section "llvm.metadata"

// PARAM: private unnamed_addr constant [12 x i8] c"param_ann_{{.}}\00", section "llvm.metadata"
// PARAM: private unnamed_addr constant [12 x i8] c"param_ann_{{.}}\00", section "llvm.metadata"
// PARAM: private unnamed_addr constant [12 x i8] c"param_ann_{{.}}\00", section "llvm.metadata"
// PARAM: private unnamed_addr constant [12 x i8] c"param_ann_{{.}}\00", section "llvm.metadata"

int foo(int v __attribute__((annotate("param_ann_2"))) __attribute__((annotate("param_ann_3"))));
int foo(int v __attribute__((annotate("param_ann_0"))) __attribute__((annotate("param_ann_1")))) {
    return v + 1;
// PARAM: define {{.*}}@foo
// PARAM:      [[V:%.*]] = alloca i32
// PARAM:      bitcast i32* [[V]] to i8*
// PARAM-NEXT: call void @llvm.var.annotation(
// PARAM-NEXT: bitcast i32* [[V]] to i8*
// PARAM-NEXT: call void @llvm.var.annotation(
// PARAM-NEXT: bitcast i32* [[V]] to i8*
// PARAM-NEXT: call void @llvm.var.annotation(
// PARAM-NEXT: bitcast i32* [[V]] to i8*
// PARAM-NEXT: call void @llvm.var.annotation(
}

void local(void) {
    int localvar __attribute__((annotate("localvar_ann_0"))) __attribute__((annotate("localvar_ann_1"))) = 3;
// LOCAL-LABEL: define void @local()
// LOCAL:      [[LOCALVAR:%.*]] = alloca i32,
// LOCAL-NEXT: [[T0:%.*]] = bitcast i32* [[LOCALVAR]] to i8*
// LOCAL-NEXT: call void @llvm.var.annotation(i8* [[T0]], i8* getelementptr inbounds ([15 x i8], [15 x i8]* @{{.*}}), i8* getelementptr inbounds ({{.*}}), i32 33, i8* null)
// LOCAL-NEXT: [[T0:%.*]] = bitcast i32* [[LOCALVAR]] to i8*
// LOCAL-NEXT: call void @llvm.var.annotation(i8* [[T0]], i8* getelementptr inbounds ([15 x i8], [15 x i8]* @{{.*}}), i8* getelementptr inbounds ({{.*}}), i32 33, i8* null)
}

void local_after_return(void) {
    return;
    int localvar __attribute__((annotate("localvar_after_return"))) = 3;
// Test we are not emitting instructions like bitcast or call outside of a basic block.
// LOCAL-LABEL: define void @local_after_return()
// LOCAL:      [[LOCALVAR:%.*]] = alloca i32,
// LOCAL-NEXT: ret void
}

void undef(void) {
    int undefvar __attribute__((annotate("undefvar_ann_0")));
// UNDEF-LABEL: define void @undef()
// UNDEF:      [[UNDEFVAR:%.*]] = alloca i32,
// UNDEF-NEXT: [[T0:%.*]] = bitcast i32* [[UNDEFVAR]] to i8*
// UNDEF-NEXT: call void @llvm.var.annotation(i8* [[T0]], i8* getelementptr inbounds ([15 x i8], [15 x i8]* @{{.*}}), i8* getelementptr inbounds ({{.*}}), i32 52, i8* null)
}
