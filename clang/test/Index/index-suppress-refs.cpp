
#include "index-suppress-refs.hpp"

class Sub : B1, B2 {
  typedef B1 Base1;
  typedef B2 Base2;
};

// RUN: env CINDEXTEST_SUPPRESSREFS=1 c-index-test -index-file %s | FileCheck %s
// CHECK:      [indexDeclaration]: kind: c++-class | name: Sub
// CHECK-NOT:  [indexEntityReference]: kind: c++-class | name: B1
// CHECK-NOT:  [indexEntityReference]: kind: c++-class | name: B2
