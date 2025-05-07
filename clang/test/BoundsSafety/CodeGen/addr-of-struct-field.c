
// RUN: %clang_cc1 -O0  -fbounds-safety -emit-llvm %s -o /dev/null
// RUN: %clang_cc1 -O2  -fbounds-safety -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -O0  -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -emit-llvm %s -o /dev/null
// RUN: %clang_cc1 -O2  -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -emit-llvm %s -o - | FileCheck %s
struct S {
    int i;
    char *p;
    long l;
};

// CHECK-LABEL: @fails_oob
// CHECK: tail call void @llvm.ubsantrap(i8 25) #{{[0-9]+}}, !annotation ![[ANNOTATION:[0-9]+]]
// CHECK-NEXT: unreachable, !annotation ![[ANNOTATION]]
int fails_oob() {
    struct S s = {1, 0, 2};
    struct S *ps = &s;
    ps = &ps->i;
    ps = &s.i;
    return ps->i;
}
