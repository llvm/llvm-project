namespace N {
  struct A { };
  namespace M { 
    struct C { };
  };
}

namespace N {
  struct B { };
}

N::
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-1):4 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: A
// CHECK-CC1: B
// CHECK-CC1: M

