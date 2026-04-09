// Check that null pointer checks are not omited in kernel mode compilations
// RUN: %clang_cc1 -fms-kernel -fms-extensions -triple x86_64-pc-windows-msvc -O2 %s -emit-llvm -o - | FileCheck %s

// CHECK:      define dso_local i32 @process(ptr noundef readonly captures(address_is_null) %p) local_unnamed_addr #0
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = icmp eq ptr %p, null
// CHECK:      attributes #0 = {{.*}} null_pointer_is_valid

struct Obj { int value; int extra; };

int process(struct Obj* p) {
    int v = p->value;
    if (!p)
        return -1;
    return v + p->extra;
}
