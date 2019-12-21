// RUN: %clang_cc1 -fsyntax-only %s -ast-dump | FileCheck %s

struct B { _Alignas(64) struct { int b; };   };

// CHECK: AlignedAttr {{.*}} _Alignas
// CHECK: ConstantExpr {{.*}} 64
// CHECK: IntegerLiteral {{.*}} 64
