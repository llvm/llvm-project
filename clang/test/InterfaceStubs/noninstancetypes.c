// RUN: %clang_cc1 -o - -emit-interface-stubs %s | FileCheck %s
// TODO: Change clang_cc1 to clang when llvm-ifs can accept empty symbol lists.

// CHECK:      Symbols:
// CHECK-NEXT: ...

struct a;
enum { b };
typedef int c;

