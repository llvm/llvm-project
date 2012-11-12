// RUN: %clang_cc1 -ast-dump %s | FileCheck %s

template<class T>
class P {
 public:
  P(T* t) {}
};

namespace foo {
class A {};
enum B {};
typedef int C;
}

int main() {
  // CHECK: CXXNewExpr {{0x[0-9a-fA-F]+}} <col:19, col:28> 'foo::A *'
  P<foo::A> p14 = new foo::A;
  // CHECK: CXXNewExpr {{0x[0-9a-fA-F]+}} <col:19, col:28> 'foo::B *'
  P<foo::B> p24 = new foo::B;
  // CHECK: CXXNewExpr {{0x[0-9a-fA-F]+}} <col:19, col:28> 'foo::C *'
  P<foo::C> pr4 = new foo::C;
}

foo::A getName() {
  // CHECK: CXXConstructExpr {{0x[0-9a-fA-F]+}} <col:10, col:17> 'foo::A'
  return foo::A();
}
