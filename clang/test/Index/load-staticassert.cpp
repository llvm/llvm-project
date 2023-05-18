
static_assert(2 + 2 == 4, "Simple maths");

// RUN: c-index-test -test-load-source all -fno-delayed-template-parsing -std=c++11 %s | FileCheck %s
// CHECK: load-staticassert.cpp:2:1: StaticAssert=:2:1 (Definition) Extent=[2:1 - 2:42]
// CHECK: load-staticassert.cpp:2:15: BinaryOperator= Extent=[2:15 - 2:25]
// CHECK: load-staticassert.cpp:2:15: BinaryOperator= Extent=[2:15 - 2:20]
// CHECK: load-staticassert.cpp:2:15: IntegerLiteral= Extent=[2:15 - 2:16]
// CHECK: load-staticassert.cpp:2:19: IntegerLiteral= Extent=[2:19 - 2:20]
// CHECK: load-staticassert.cpp:2:24: IntegerLiteral= Extent=[2:24 - 2:25]
// CHECK: load-staticassert.cpp:2:27: StringLiteral="Simple maths" Extent=[2:27 - 2:41]
