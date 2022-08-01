// clang-format off
// REQUIRES: host-supports-jit, host-supports-exception 
// UNSUPPORTED: system-aix
//
// Tests that a global destructor is ran on platforms with gnu exception support.
//
// RUN: cat %s | clang-repl | FileCheck %s

extern "C" int printf(const char *, ...);

struct D { float f = 1.0; D *m = nullptr; D(){} ~D() { printf("D[f=%f, m=0x%llx]\n", f, reinterpret_cast<unsigned long long>(m)); }} d;
// CHECK: D[f=1.000000, m=0x0]

%quit