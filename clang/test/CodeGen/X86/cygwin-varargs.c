// RUN: %clang_cc1 -triple x86_64-windows-gnu -emit-llvm < %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-pc-cygwin -emit-llvm < %s | FileCheck %s

struct foo {
  int x;
  float y;
  char z;
};
// CHECK: %[[STRUCT_FOO:.*]] = type { i32, float, i8 }

void f(int a, ...) {
  // CHECK-LABEL: define dso_local void @f
  __builtin_va_list ap;
  __builtin_va_start(ap, a);
  // CHECK: %[[AP:.*]] = alloca ptr
  // CHECK: call void @llvm.va_start
  int b = __builtin_va_arg(ap, int);
  // CHECK: %[[AP_CUR:.*]] = load ptr, ptr %[[AP]]
  // CHECK-NEXT: %[[AP_NEXT:.*]] = getelementptr inbounds i8, ptr %[[AP_CUR]], i64 8
  // CHECK-NEXT: store ptr %[[AP_NEXT]], ptr %[[AP]]
  double _Complex c = __builtin_va_arg(ap, double _Complex);
  // CHECK: %[[AP_CUR2:.*]] = load ptr, ptr %[[AP]]
  // CHECK-NEXT: %[[AP_NEXT2:.*]] = getelementptr inbounds i8, ptr %[[AP_CUR2]], i64 8
  // CHECK-NEXT: store ptr %[[AP_NEXT2]], ptr %[[AP]]
  // CHECK-NEXT: load ptr, ptr %[[AP_CUR2]]
  struct foo d = __builtin_va_arg(ap, struct foo);
  // CHECK: %[[AP_CUR3:.*]] = load ptr, ptr %[[AP]]
  // CHECK-NEXT: %[[AP_NEXT3:.*]] = getelementptr inbounds i8, ptr %[[AP_CUR3]], i64 8
  // CHECK-NEXT: store ptr %[[AP_NEXT3]], ptr %[[AP]]
  __builtin_va_list ap2;
  __builtin_va_copy(ap2, ap);
  // CHECK: call void @llvm.va_copy
  __builtin_va_end(ap);
  // CHECK: call void @llvm.va_end
}
