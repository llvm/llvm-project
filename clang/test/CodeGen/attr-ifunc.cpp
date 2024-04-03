// RUN: %clang_cc1 -triple x86_64-linux -fsyntax-only -verify -emit-llvm-only %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx -fsyntax-only -verify -emit-llvm-only %s
// RUN: %clang_cc1 -triple arm64-apple-macosx -fsyntax-only -verify -emit-llvm-only %s
// RUN: not %clang_cc1 -triple x86_64-linux -emit-llvm-only -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -triple x86_64-apple-macosx -emit-llvm-only -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -triple arm64-apple-macosx -emit-llvm-only -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

void *f1_ifunc(void) { return nullptr; }
void f1(void) __attribute__((ifunc("f1_ifunc")));
// expected-error@-1 {{ifunc must point to a defined function}}
// expected-note@-2 {{must refer to its mangled name}}
// expected-note@-3 {{function by that name is mangled as}}
// CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:30-[[@LINE-4]]:47}:"ifunc(\"_Z8f1_ifuncv\")"

void *f6_resolver_resolver(void) { return 0; }
void *f6_resolver(void) __attribute__((ifunc("f6_resolver_resolver")));
// expected-error@-1 {{ifunc must point to a defined function}}
// expected-note@-2 {{must refer to its mangled name}}
// expected-note@-3 {{function by that name is mangled as}}
// CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:40-[[@LINE-4]]:69}:"ifunc(\"_Z20f6_resolver_resolverv\")"
void f6(void) __attribute__((ifunc("f6_resolver")));
// expected-error@-1 {{ifunc must point to a defined function}}
// expected-note@-2 {{must refer to its mangled name}}
// expected-note@-3 {{function by that name is mangled as}}
// CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:30-[[@LINE-4]]:50}:"ifunc(\"_Z11f6_resolverv\")"

__attribute__((unused, ifunc("resolver"), deprecated("hahahaha, isn't C great?")))
void func();
// expected-error@-2 {{ifunc must point to a defined function}}
// expected-note@-3 {{must refer to its mangled name}}

