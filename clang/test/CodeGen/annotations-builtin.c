// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm -o - %s | FileCheck %s
// END.

static long long llfoo;
static int intfoo;
static short shortfoo;
static char charfoo;

// CHECK: private unnamed_addr constant [13 x i8] {{.*}}annotation_a{{.*}} section "llvm.metadata"
// CHECK-NOT: {{.*}}annotation_a{{.*}}

static int foo(int a) {
    return a + 1;
}

int main(int argc, char **argv) {
    char barray[16];
    char *b = (char *) __builtin_annotation((int)barray, "annotation_a");
// CHECK: ptrtoint i8* {{.*}} to i32
// CHECK-NEXT: call i32 @llvm.annotation.i32
// CHECK: inttoptr {{.*}} to i8*

    int call = __builtin_annotation(foo(argc), "annotation_a");
// CHECK: call {{.*}} @foo
// CHECK: call i32 @llvm.annotation.i32

    long long lla = __builtin_annotation(llfoo, "annotation_a");
// CHECK: call i64 @llvm.annotation.i64

    int inta = __builtin_annotation(intfoo, "annotation_a");
// CHECK: load i32, i32* @intfoo
// CHECK-NEXT: call i32 @llvm.annotation.i32
// CHECK-NEXT: store

    short shorta =  __builtin_annotation(shortfoo, "annotation_a");
// CHECK: call i16 @llvm.annotation.i16

    char chara = __builtin_annotation(charfoo, "annotation_a");
// CHECK: call i8 @llvm.annotation.i8

    char **arg = (char**) __builtin_annotation((int) argv, "annotation_a");
// CHECK: ptrtoint i8** {{.*}} to
// CHECK: call i32 @llvm.annotation.i32
// CHECK: inttoptr {{.*}} to i8**
    return 0;

    int after_return = __builtin_annotation(argc, "annotation_a");
// CHECK-NOT: call i32 @llvm.annotation.i32
}
