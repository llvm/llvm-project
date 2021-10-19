// Check that clang can use a PCH created from libclang.

// https://PR46644
// XFAIL: arm64-apple

// This test doesn't use -fdisable-module-hash and hence requires that
// CompilerInvocation::getModuleHash() computes exactly the same hash
// for c-index-test and clang, which in turn requires that the both use
// exactly the same resource-dir, even without calling realpath() on it:
// - a/../b/ and b/ are not considered the same
// - on Windows, c:\ and C:\ (only different in case) are not the same

// RUN: rm -rf %t.mcp %t.h.pch
// RUN: %clang_cc1 -fsyntax-only %s -verify
// RUN: c-index-test -write-pch %t.h.pch %s -fmodules -fmodules-cache-path=%t.mcp -Xclang -triple -Xclang x86_64-apple-darwin
// RUN: %clang -fsyntax-only -include %t.h %s -Xclang -verify -fmodules -fmodules-cache-path=%t.mcp -Xclang -detailed-preprocessing-record -Xclang -triple -Xclang x86_64-apple-darwin -Xclang -fallow-pch-with-compiler-errors
// RUN: %clang -x c-header %s -o %t.clang.h.pch -fmodules -fmodules-cache-path=%t.mcp -Xclang -detailed-preprocessing-record -Xclang -triple -Xclang x86_64-apple-darwin -Xclang -fallow-pch-with-compiler-errors -Xclang -verify
// RUN: c-index-test -test-load-source local %s -include %t.clang.h -fmodules -fmodules-cache-path=%t.mcp -Xclang -triple -Xclang x86_64-apple-darwin | FileCheck %s

// FIXME: Still fails on at least some linux boxen.
// REQUIRES: system-darwin

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
