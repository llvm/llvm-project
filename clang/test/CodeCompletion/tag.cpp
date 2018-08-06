class X { };
struct Y { };

namespace N {
  template<typename> class Z;
}

namespace M {
  class A;
}
using M::A;

namespace N {
  class Y;
  
  void test() {
    class 
    // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:17:11 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
    // FIXME: the redundant Y is really annoying... it needs qualification to 
    // actually be useful. Here, it just looks redundant :(
    // CHECK-CC1: A
    // CHECK-CC1: M : M::
    // CHECK-CC1: N : N::
    // CHECK-CC1: X
    // CHECK-CC1: Y
    // CHECK-CC1: Y
    // CHECK-CC1: Z
