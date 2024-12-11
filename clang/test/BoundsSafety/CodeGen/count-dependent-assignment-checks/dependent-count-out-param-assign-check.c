

// RUN: %clang_cc1 -O0  -fbounds-safety -emit-llvm %s -o /dev/null
// RUN: %clang_cc1 -O2  -fbounds-safety -emit-llvm %s -o - | FileCheck %s
#include <ptrcheck.h>

int foo(int *__counted_by(*out_len)* out_buf, int *out_len) {
    int arr[10];
    int n = 11;
    *out_len = n; // trap : 11 > boundof(arr)
    *out_buf = arr;
    return 0;
}

int bar(int *__counted_by(*out_len)* out_buf, int *out_len) {
    int arr[10];
    *out_len = 9;
    *out_buf = arr;
    return 0;
}

int main() {
    int len;
    int *__counted_by(len) buf;
    foo(&buf, &len);

    return buf[len-1];
}

// CHECK: define noundef i32 @foo
// CHECK: {{.*}}:
// CHECK:   call void @llvm.ubsantrap(i8 25)
// CHECK:   unreachable

// CHECK: define noundef i32 @bar
// CHECK: {{.*}}:
// CHECK:   ret i32 0

// CHECK: define noundef i32 @main()
// CHECK: {{.*}}:
// CHECK:   tail call void @llvm.ubsantrap(i8 25)
// CHECK:   unreachable
