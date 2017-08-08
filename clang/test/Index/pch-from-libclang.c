// Check that clang can use a PCH created from libclang.

// FIXME: Non-darwin bots fail. Would need investigation using -module-file-info to see what is the difference in modules generated from libclang vs the compiler invocation, in those systems.
// REQUIRES: system-darwin

// RUN: %clang_cc1 -fsyntax-only %s -verify
// RUN: c-index-test -write-pch %t.h.pch %s -fmodules -fmodules-cache-path=%t.mcp -Xclang -triple -Xclang x86_64-apple-darwin
// RUN: %clang -fsyntax-only -include %t.h %s -Xclang -verify -fmodules -fmodules-cache-path=%t.mcp -Xclang -detailed-preprocessing-record -Xclang -triple -Xclang x86_64-apple-darwin -Xclang -fallow-pch-with-compiler-errors
// RUN: %clang -x c-header %s -o %t.clang.h.pch -fmodules -fmodules-cache-path=%t.mcp -Xclang -detailed-preprocessing-record -Xclang -triple -Xclang x86_64-apple-darwin -Xclang -fallow-pch-with-compiler-errors -Xclang -verify
// RUN: c-index-test -test-load-source local %s -include %t.clang.h -fmodules -fmodules-cache-path=%t.mcp -Xclang -triple -Xclang x86_64-apple-darwin | FileCheck %s

#ifndef HEADER
#define HEADER

void some_function(undeclared_type p); // expected-error{{unknown type name}}

struct S { int x; };

#else
// expected-no-diagnostics

void test(struct S *s) {
  // CHECK: [[@LINE+1]]:6: MemberRefExpr=x:[[@LINE-6]]:16
  s->x = 0;
}

#endif
