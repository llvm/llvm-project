// RUN: %clang_cc1 -triple aarch64-pc-windows-msvc -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-pc-windows-msvc -O2 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-pc-windows-msvc -fsanitize=thread -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=SAN
// RUN: %clang_cc1 -triple aarch64-pc-windows-msvc -fsanitize=address -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=SAN

// REQUIRES: aarch64-registered-target

int foo(int) __attribute__ ((ifunc("foo_ifunc")));

static int f1(int i) {
  return i + 1;
}

static int f2(int i) {
  return i + 2;
}

typedef int (*foo_t)(int);

volatile int global;

static foo_t foo_ifunc(void) {
  return global ? f1 : f2;
}

int bar(void) {
  return foo(1);
}

extern void goo(void);

void bar2(void) {
  goo();
}

extern void goo(void) __attribute__ ((ifunc("goo_ifunc")));

void* goo_ifunc(void) {
  return 0;
}

/// The ifunc is emitted after its resolver.
void *hoo_ifunc(void) { return 0; }
extern void hoo(int) __attribute__ ((ifunc("hoo_ifunc")));

/// ifunc on Windows is lowered to global pointers and an indirect call.
// CHECK: @global = dso_local global i32 0, align 4
// CHECK: {{.*}} = internal{{.*}}global{{.*}}poison, align 8
/// Register the constructor for initialisation.
// CHECK: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 10, ptr @{{.*}}, ptr null }]

// CHECK-LABEL: @bar()
// CHECK   %0 = load ptr, ptr @0, align 8
// CHECK   %call = call i32 %0(i32 noundef 1)

// CHECK-LABEL: @bar2()
// CHECK %0 = load ptr, ptr getelementptr inbounds ([3 x ptr], ptr @0, i32 0, i32 1), align 8
// CHECK call void %0()

// CHECK: define internal void @{{.*}}()

// SAN: define {{(dso_local )?}}noalias {{(noundef )?}}ptr @goo_ifunc() {{(local_unnamed_addr )?}}

// SAN: define {{(dso_local )?}}noalias {{(noundef )?}}ptr @hoo_ifunc() {{(local_unnamed_addr )?}}

// SAN: define internal {{(fastcc )?}}{{(noundef )?}}nonnull ptr @foo_ifunc() {{(unnamed_addr )?}}

