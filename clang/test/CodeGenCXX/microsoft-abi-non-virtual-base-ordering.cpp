// RUN: %clang_cc1 -fno-rtti -emit-llvm -triple=i686-pc-win32 -o - %s  2>/dev/null | FileCheck %s
// RUN: %clang_cc1 -fno-rtti -emit-llvm -triple=x86_64-pc-win32 -o - %s  2>/dev/null | FileCheck %s

struct C0 { int a; };
struct C1 { int a; virtual void C1M() {} };
struct C2 { int a; virtual void C2M() {} };
struct C3 : C0, C1, C2 {} a;

// Check to see that both C1 and C2 get laid out before C0 does.
// CHECK: %struct.C3 = type { %struct.C1, %struct.C2, %struct.C0 }
