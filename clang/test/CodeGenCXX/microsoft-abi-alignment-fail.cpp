// RUN: %clang_cc1 -fno-rtti -emit-llvm -triple=i686-pc-win32 -o - %s  2>/dev/null | FileCheck %s
// RUN: %clang_cc1 -fno-rtti -emit-llvm -triple=x86_64-pc-win32 -o - %s  2>/dev/null | FileCheck %s -check-prefix CHECK-X64

struct B { char a; };
struct A : virtual B {} a;

// The <> indicate that the pointer is packed, which is required to support
// microsoft layout in 32 bit mode, but not 64 bit mode.
// CHECK: %struct.A = type <{ ptr, %struct.B }>
// CHECK-X64: %struct.A = type { ptr, %struct.B }
