// Check that null pointer checks are not optimized out if -fms-compatibility is set
// RUN: %clang_cc1 -fms-kernel -O2 -triple x86_64-pc-windows-msvc %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -O2 -triple x86_64-pc-windows-msvc %s -emit-llvm -o - | FileCheck %s --check-prefix=NOCHECK
// RUN: %clang_cc1 -fms-kernel -O2 -triple x86_64-pc-windows-msvc -fdelete-null-pointer-checks %s -emit-llvm -o - | FileCheck %s --check-prefix=NOCHECK
// RUN: %clang_cc1 -O2 -triple x86_64-pc-windows-msvc -fdelete-null-pointer-checks -fms-kernel %s -emit-llvm -o - | FileCheck %s --check-prefix=NOCHECK

// CHECK-LABEL: i32 @process
// CHECK-NEXT:  entry:
// CHECK-NEXT:    %tobool.not = icmp eq ptr %p, null
// CHECK-NEXT:    br i1 %tobool.not, label %cleanup, label %if.end

// CHECK-LABEL: ptr @call_memcpy
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.memcpy.p0.p0.i64
// CHECK-NEXT:    ret ptr null


// NOCHECK-LABEL: i32 @process
// NOCHECK-NOT:   icmp eq ptr %p, null

// NOCHECK-LABEL: ptr @call_memcpy
// NOCHECK-NEXT:  entry:
// NOCHECK-NEXT:    ret ptr null

struct Obj { int value; int extra; };

int process(struct Obj* p) {
    int v = p->value;
    if (!p)
        return -1;
    return v + p->extra;
}

void* call_memcpy(void* p, long long size) {
  return __builtin_memcpy(0, p, size);
}
