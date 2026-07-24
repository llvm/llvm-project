// clang-format off
//
// Regression test for https://github.com/llvm/llvm-project/issues/186534
// This is based on global-dtor.cpp, but has many instances of 'struct D',
// which makes it easier to reproduce the issue.
//
// RUN: cat %s | clang-repl | FileCheck %s

extern "C" int printf(const char *, ...);

struct D { float f = 1.0; D *m = nullptr; D(){} ~D() { printf("D[f=%f, m=0x%llx]\n", f, reinterpret_cast<unsigned long long>(m)); }};

struct D d1;
struct D d2;
struct D d3;
struct D d4;
struct D d5;
struct D d6;
struct D d7;
struct D d8;
struct D d9;
struct D d10;
struct D d11;
struct D d12;
struct D d13;
struct D d14;
struct D d15;
struct D d16;
struct D d17;
struct D d18;
struct D d19;
struct D d20;
struct D d21;
struct D d22;
struct D d23;
struct D d24;
struct D d25;
struct D d26;
struct D d27;
struct D d28;
struct D d29;
struct D d30;

// CHECK: D[f=1.000000, m=0x0]
// CHECK: D[f=1.000000, m=0x0]
// CHECK: D[f=1.000000, m=0x0]
// CHECK: D[f=1.000000, m=0x0]
// CHECK: D[f=1.000000, m=0x0]
// CHECK: D[f=1.000000, m=0x0]
// CHECK: D[f=1.000000, m=0x0]
// CHECK: D[f=1.000000, m=0x0]
// CHECK: D[f=1.000000, m=0x0]
// CHECK: D[f=1.000000, m=0x0]
// CHECK: D[f=1.000000, m=0x0]
// CHECK: D[f=1.000000, m=0x0]
// CHECK: D[f=1.000000, m=0x0]
// CHECK: D[f=1.000000, m=0x0]
// CHECK: D[f=1.000000, m=0x0]
// CHECK: D[f=1.000000, m=0x0]
// CHECK: D[f=1.000000, m=0x0]
// CHECK: D[f=1.000000, m=0x0]
// CHECK: D[f=1.000000, m=0x0]
// CHECK: D[f=1.000000, m=0x0]
// CHECK: D[f=1.000000, m=0x0]
// CHECK: D[f=1.000000, m=0x0]
// CHECK: D[f=1.000000, m=0x0]
// CHECK: D[f=1.000000, m=0x0]
// CHECK: D[f=1.000000, m=0x0]
// CHECK: D[f=1.000000, m=0x0]
// CHECK: D[f=1.000000, m=0x0]
// CHECK: D[f=1.000000, m=0x0]
// CHECK: D[f=1.000000, m=0x0]
// CHECK: D[f=1.000000, m=0x0]

%quit
