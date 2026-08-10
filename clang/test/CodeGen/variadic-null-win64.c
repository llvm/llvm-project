// RUN: %clang_cc1 %s -emit-llvm -o - -triple x86_64-windows-msvc | FileCheck %s --check-prefix=WINDOWS
// RUN: %clang_cc1 %s -emit-llvm -o - -triple x86_64-linux | FileCheck %s --check-prefix=LINUX

// Make it possible to pass NULL through variadic functions on platforms where
// NULL has an integer type that is more narrow than a pointer. On such
// platforms we widen null pointer constants passed to variadic functions to a
// pointer-sized integer. We don't apply this special case to K&R-style
// unprototyped functions, because MSVC doesn't either.

#define NULL 0

void v(const char *f, ...);
void kr();
void f(const char *f) {
  v(f, 1, 2, 3, NULL);
  kr(f, 1, 2, 3, 0);
}
// WINDOWS: define dso_local void @f(ptr noundef %f)
// WINDOWS: call void (ptr, ...) @v(ptr {{.*}}, i32 noundef 1, i32 noundef 2, i32 noundef 3, i64 noundef 0)
// WINDOWS: call void @kr(ptr noundef {{.*}}, i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 0)
// LINUX: define{{.*}} void @f(ptr noundef %f)
// LINUX: call void (ptr, ...) @v(ptr {{.*}}, i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 0)
// LINUX: call void (ptr, i32, i32, i32, i32, ...) @kr(ptr {{.*}}, i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 0)
