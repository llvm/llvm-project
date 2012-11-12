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
// CHECK: getelementptr inbounds %struct.foo* %f, i32 0, i32 0
// CHECK-NEXT: bitcast i32* {{.*}} to i8*
// CHECK-NEXT: call i8* @llvm.ptr.annotation.p0i8({{.*}}str{{.*}}str{{.*}}i32 8)
// CHECK-NEXT: bitcast i8* {{.*}} to i32*
// CHECK-NEXT: bitcast i32* {{.*}} to i8*
// CHECK-NEXT: call i8* @llvm.ptr.annotation.p0i8({{.*}}str{{.*}}str{{.*}}i32 8)
// CHECK-NEXT: bitcast i8* {{.*}} to i32*
    gf.v = argc;
// CHECK: bitcast i32* getelementptr inbounds (%struct.foo* @gf, i32 0, i32 0) to i8*
// CHECK-NEXT: call i8* @llvm.ptr.annotation.p0i8({{.*}}str{{.*}}str{{.*}}i32 8)
    return 0;
}
