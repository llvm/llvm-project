// Check that clang can use a PCH created from libclang.

// FIXME: Non-darwin bots fail. Would need investigation using -module-file-info to see what is the difference in modules generated from libclang vs the compiler invocation, in those systems.
// REQUIRES: system-darwin

// RUN: c-index-test -write-pch %t.h.pch %s -fmodules -fmodules-cache-path=%t.mcp -Xclang -triple -Xclang x86_64-apple-darwin
// RUN: %clang -fsyntax-only -include %t.h %s -Xclang -verify -fmodules -fmodules-cache-path=%t.mcp -Xclang -detailed-preprocessing-record -Xclang -triple -Xclang x86_64-apple-darwin

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

struct S { int x; };

#else

void test(struct S *s) {
  s->x = 0;
}

#endif
