// Test without serialization:
// RUN: %clang_cc1 -fsyntax-only %s -ast-dump | FileCheck %s
//
// Test with serialization:
// RUN: %clang_cc1 -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -include-pch %t -ast-dump-all /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck %s

struct B { _Alignas(64) struct { int b; };   };

// CHECK: AlignedAttr {{.*}} _Alignas
// CHECK: ConstantExpr {{.*}} 64
// CHECK: IntegerLiteral {{.*}} 64
