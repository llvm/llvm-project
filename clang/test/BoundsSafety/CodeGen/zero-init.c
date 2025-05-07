
// RUN: %clang_cc1 %s -O2  -fbounds-safety -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -O2  -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -emit-llvm -o - | FileCheck %s

int main() {
    int *ptr;
    if (!ptr)
        return 0;
    return 1;
}

// CHECK: ret i32 0
// CHECK-NOT ret i32 1
