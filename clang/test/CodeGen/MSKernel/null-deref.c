// Check that null pointer checks are not omited in kernel mode compilations
// RUN: %clang_cc1 -fms-kernel -fms-extensions -triple x86_64-pc-windows-msvc %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -fms-kernel -fms-extensions -triple x86_64-pc-windows-msvc -fdelete-null-pointer-checks %s -emit-llvm -o - | FileCheck %s --check-prefix=NOCHECK

// CHECK: define dso_local i32 @process(ptr noundef %p) #0
// CHECK: attributes #0 = {{.*}} null_pointer_is_valid
// NOCHECK-NOT: null_pointer_is_valid

struct Obj { int value; int extra; };

int process(struct Obj* p) {
    int v = p->value;
    if (!p)
        return -1;
    return v + p->extra;
}
