// RUN: %clang_cc1 -triple i386-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple i386-unknown-linux-gnu -O2 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple i386-unknown-linux-gnu -fsanitize=thread -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=SAN
// RUN: %clang_cc1 -triple i386-unknown-linux-gnu -fsanitize=address -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=SAN
// RUN: %clang_cc1 -triple i386-unknown-linux-gnu -fsanitize=memory -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=SAN
// RUN: %clang_cc1 -triple arm64-apple-macosx -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-macosx -O2 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx -O2 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-macosx -fsanitize=thread -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=MACSAN
// RUN: %clang_cc1 -triple x86_64-apple-macosx -fsanitize=thread -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=MACSAN
// RUN: %clang_cc1 -triple arm64-apple-macosx -fsanitize=address -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=MACSAN
// RUN: %clang_cc1 -triple x86_64-apple-macosx -fsanitize=address -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=MACSAN

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
// CHECK: @foo = ifunc i32 (i32), ptr @foo_ifunc
// CHECK: @goo = ifunc void (), ptr @goo_ifunc

// CHECK: call i32 @foo(i32
// CHECK: call void @goo()

// SAN: define internal nonnull {{(noundef )?}}ptr @foo_ifunc() #[[#FOO_IFUNC:]] {
// MACSAN: define internal nonnull {{(noundef )?}}ptr @foo_ifunc() #[[#FOO_IFUNC:]] {

// SAN: define dso_local noalias {{(noundef )?}}ptr @goo_ifunc() #[[#GOO_IFUNC:]] {
// MACSAN: define noalias {{(noundef )?}}ptr @goo_ifunc() #[[#GOO_IFUNC:]] {

// SAN-DAG: attributes #[[#FOO_IFUNC]] = {{{.*}} disable_sanitizer_instrumentation {{.*}}
// MACSAN-DAG: attributes #[[#FOO_IFUNC]] = {{{.*}} disable_sanitizer_instrumentation {{.*}}

// SAN-DAG: attributes #[[#GOO_IFUNC]] = {{{.*}} disable_sanitizer_instrumentation {{.*}}
// MACSAN-DAG: attributes #[[#GOO_IFUNC]] = {{{.*}} disable_sanitizer_instrumentation {{.*}}
