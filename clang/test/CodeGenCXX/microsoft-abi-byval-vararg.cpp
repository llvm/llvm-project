// RUN: %clang_cc1 -no-opaque-pointers -Wno-non-pod-varargs -emit-llvm %s -o - -triple=i686-pc-win32 -mconstructor-aliases -fno-rtti | FileCheck %s

#include <stdarg.h>

struct A {
  A(int a) : a(a) {}
  A(const A &o) : a(o.a) {}
  ~A() {}
  int a;
};

int foo(A a, ...) {
  va_list ap;
  va_start(ap, a);
  int sum = 0;
  for (int i = 0; i < a.a; ++i)
    sum += va_arg(ap, int);
  va_end(ap);
  return sum;
}

// CHECK-LABEL: define dso_local noundef i32 @"?foo@@YAHUA@@ZZ"(<{ %struct.A }>* inalloca(<{ %struct.A }>) %0, ...)

int main() {
  return foo(A(3), 1, 2, 3);
}
// CHECK-LABEL: define dso_local noundef i32 @main()
// CHECK: %[[argmem:[^ ]*]] = alloca inalloca <{ %struct.A, i32, i32, i32 }>
// CHECK: call noundef i32 {{.*bitcast.*}}@"?foo@@YAHUA@@ZZ"{{.*}}(<{ %struct.A, i32, i32, i32 }>* inalloca(<{ %struct.A, i32, i32, i32 }>) %[[argmem]])

void varargs_zero(...);
void varargs_one(int, ...);
void varargs_two(int, int, ...);
void varargs_three(int, int, int, ...);
void call_var_args() {
  A x(3);
  varargs_zero(x);
  varargs_one(1, x);
  varargs_two(1, 2, x);
  varargs_three(1, 2, 3, x);
}

// CHECK-LABEL: define dso_local void @"?call_var_args@@YAXXZ"()
// CHECK: call void {{.*bitcast.*varargs_zero.*}}(<{ %struct.A }>* inalloca(<{ %struct.A }>) %{{.*}})
// CHECK: call void {{.*bitcast.*varargs_one.*}}(<{ i32, %struct.A }>* inalloca(<{ i32, %struct.A }>) %{{.*}})
// CHECK: call void {{.*bitcast.*varargs_two.*}}(<{ i32, i32, %struct.A }>* inalloca(<{ i32, i32, %struct.A }>) %{{.*}})
// CHECK: call void {{.*bitcast.*varargs_three.*}}(<{ i32, i32, i32, %struct.A }>* inalloca(<{ i32, i32, i32, %struct.A }>) %{{.*}})

// CHECK-LABEL: declare dso_local void @"?varargs_zero@@YAXZZ"(...)
// CHECK-LABEL: declare dso_local void @"?varargs_one@@YAXHZZ"(i32 noundef, ...)
// CHECK-LABEL: declare dso_local void @"?varargs_two@@YAXHHZZ"(i32 noundef, i32 noundef, ...)
// CHECK-LABEL: declare dso_local void @"?varargs_three@@YAXHHHZZ"(i32 noundef, i32 noundef, i32 noundef, ...)
