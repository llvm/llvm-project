// clang-format off
// FIXME: Merge into global-dtor.cpp when exception support arrives on windows-msvc
// REQUIRES: host-supports-jit && windows-msvc
//
// Tests that a global destructor is ran in windows-msvc platform.
//
// RUN: cat %s | clang-repl | FileCheck %s

extern "C" int printf(const char *, ... );

struct D { float f = 1.0; D *m = nullptr; D(){} ~D() { printf("D[f=%f, m=0x%llx]\n", f, reinterpret_cast<unsigned long long>(m)); }} d;
// CHECK: D[f=1.000000, m=0x0]

%quit