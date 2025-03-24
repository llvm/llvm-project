// RUN: %clang_cc1 -triple x86_64-linux -verify -emit-llvm-only -DERR %s
// RUN: not %clang_cc1 -triple x86_64-linux -emit-llvm-only -fdiagnostics-parseable-fixits -DERR %s 2>&1 | FileCheck %s --check-prefix=FIXIT
// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm %s -o - | FileCheck %s

#ifdef ERR
void *f1_ifunc(void) { return nullptr; }
void f1(void) __attribute__((alias("f1_ifunc")));
// expected-error@-1 {{alias must point to a defined variable or function}}
// expected-note@-2 {{must refer to its mangled name}}
// expected-note@-3 {{function by that name is mangled as}}
// FIXIT: fix-it:"{{.*}}":{[[@LINE-4]]:30-[[@LINE-4]]:47}:"alias(\"_Z8f1_ifuncv\")"

void *f6_resolver_resolver(void) { return 0; }
void *f6_resolver(void) __attribute__((alias("f6_resolver_resolver")));
// expected-error@-1 {{alias must point to a defined variable or function}}
// expected-note@-2 {{must refer to its mangled name}}
// expected-note@-3 {{function by that name is mangled as}}
// FIXIT: fix-it:"{{.*}}":{[[@LINE-4]]:40-[[@LINE-4]]:69}:"alias(\"_Z20f6_resolver_resolverv\")"
void f6(void) __attribute__((alias("f6_resolver")));
// expected-error@-1 {{alias must point to a defined variable or function}}
// expected-note@-2 {{must refer to its mangled name}}
// expected-note@-3 {{function by that name is mangled as}}
// FIXIT: fix-it:"{{.*}}":{[[@LINE-4]]:30-[[@LINE-4]]:50}:"alias(\"_Z11f6_resolverv\")"

__attribute__((unused, alias("resolver"), deprecated("hahahaha, isn't C great?")))
void func();
// expected-error@-2 {{alias must point to a defined variable or function}}
// expected-note@-3 {{must refer to its mangled name}}

void *operator new(unsigned long) __attribute__((alias("A"))); // expected-error {{alias must point to a defined variable or function}} \
                                                               // expected-note {{the function or variable specified in an alias must refer to its mangled name}}
#endif

// CHECK: @_ZN4libc4log2Ed ={{.*}} alias double (double), ptr @log2
// CHECK: define{{.*}} @log2(
namespace libc { double log2(double x); }
extern "C" double log2(double);
namespace std { using ::log2; }
using std::log2;

namespace libc {
decltype(libc::log2) __log2_impl__ __asm__("log2");
decltype(libc::log2) log2 [[gnu::alias("log2")]];
double __log2_impl__(double x) { return x; }
}
