// RUN: %clang_cc1 -no-enable-noundef-analysis %s -o - -emit-llvm | FileCheck %s
// XFAIL: target={{(aarch64|arm64).*}}, target=x86_64-pc-windows-msvc, target=x86_64-{{(pc|w64)}}-windows-gnu, target=x86_64-pc-windows-cygnus

// PR1513

// AArch64 ABI actually requires the reverse of what this is testing: the callee
// does any extensions and remaining bits are unspecified.

// Win64 ABI does expect extensions for type smaller than 64bits.

// Technically this test wasn't written to test that feature, but it's a
// valuable check nevertheless.

struct s{
long a;
long b;
};

void f(struct s a, char *b, signed char C) {
  // CHECK: i8 signext

}
