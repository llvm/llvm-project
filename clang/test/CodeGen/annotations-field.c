// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s
// END.

// CHECK: private unnamed_addr constant [8 x i8] c"v_ann_{{.}}\00", section "llvm.metadata"
// CHECK: private unnamed_addr constant [8 x i8] c"v_ann_{{.}}\00", section "llvm.metadata"

struct foo {
    int v __attribute__((annotate("v_ann_0"))) __attribute__((annotate("v_ann_1")));
};

static struct foo gf;

int main(int argc, char **argv) {
    struct foo f;
    f.v = argc;
// CHECK: getelementptr inbounds %struct.foo, ptr %f, i32 0, i32 0
// CHECK-NEXT: call ptr @llvm.ptr.annotation.p0({{.*}}str{{.*}}str{{.*}}i32 8, ptr null)
// CHECK-NEXT: call ptr @llvm.ptr.annotation.p0({{.*}}str{{.*}}str{{.*}}i32 8, ptr null)
    gf.v = argc;
// CHECK: call ptr @llvm.ptr.annotation.p0(ptr @gf, {{.*}}str{{.*}}str{{.*}}i32 8, ptr null)
    return 0;
}
