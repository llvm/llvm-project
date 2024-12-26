// RUN: %clang_cc1 -triple i386-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple i386-unknown-linux-gnu -O2 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple i386-unknown-linux-gnu -fsanitize=thread -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=SAN
// RUN: %clang_cc1 -triple i386-unknown-linux-gnu -fsanitize=address -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=SAN
// RUN: %clang_cc1 -triple i386-unknown-linux-gnu -fsanitize=memory -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=SAN
// RUN: %clang_cc1 -triple arm64-apple-macosx -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-macosx -O2 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx -O2 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-macosx -fsanitize=thread -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=SAN
// RUN: %clang_cc1 -triple x86_64-apple-macosx -fsanitize=thread -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=SAN
// RUN: %clang_cc1 -triple arm64-apple-macosx -fsanitize=address -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=SAN
// RUN: %clang_cc1 -triple x86_64-apple-macosx -fsanitize=address -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=SAN
// RUN: %clang_cc1 -triple avr-unknown-unknown -emit-llvm -o - %s | FileCheck %s --check-prefix=AVR
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -O2 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -fsanitize=thread -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=SAN
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -fsanitize=address -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=SAN


/// The ifunc is emitted before its resolver.
int foo(int) __attribute__ ((ifunc("foo_ifunc")));

static int f1(int i) {
  return i + 1;
}

static int f2(int i) {
  return i + 2;
}

typedef int (*foo_t)(int);

int global;

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

// CHECK: @foo = ifunc i32 (i32), ptr @foo_ifunc
// CHECK: @goo = ifunc void (), ptr @goo_ifunc
// CHECK: @hoo = ifunc void (i32), ptr @hoo_ifunc

// AVR: @foo = ifunc i16 (i16), ptr addrspace(1) @foo_ifunc
// AVR: @goo = ifunc void (), ptr addrspace(1) @goo_ifunc
// AVR: @hoo = ifunc void (i16), ptr addrspace(1) @hoo_ifunc

// CHECK: call i32 @foo(i32
// CHECK: call void @goo()

// SAN: define {{(dso_local )?}}noalias {{(noundef )?}}ptr @goo_ifunc() #[[#GOO_IFUNC:]] {

// SAN: define {{(dso_local )?}}noalias {{(noundef )?}}ptr @hoo_ifunc() #[[#GOO_IFUNC]] {

// SAN: define internal {{(noundef )?}}nonnull ptr @foo_ifunc() #[[#FOO_IFUNC:]] {

// SAN-DAG: attributes #[[#GOO_IFUNC]] = {{{.*}} disable_sanitizer_instrumentation {{.*}}
// SAN-DAG: attributes #[[#FOO_IFUNC]] = {{{.*}} disable_sanitizer_instrumentation {{.*}}
