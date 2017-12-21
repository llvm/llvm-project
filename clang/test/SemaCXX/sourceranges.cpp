// RUN: %clang_cc1 -triple i686-mingw32 -ast-dump %s | FileCheck %s

template<class T>
class P {
 public:
  P(T* t) {}
};

namespace foo {
class A { public: A(int = 0) {} };
enum B {};
typedef int C;
}

// CHECK: VarDecl {{0x[0-9a-fA-F]+}} <line:16:1, col:36> col:15 ImplicitConstrArray 'foo::A [2]'
static foo::A ImplicitConstrArray[2];

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

void destruct(foo::A *a1, foo::A *a2, P<int> *p1) {
  // CHECK: MemberExpr {{0x[0-9a-fA-F]+}} <col:3, col:8> '<bound member function type>' ->~A
  a1->~A();
  // CHECK: MemberExpr {{0x[0-9a-fA-F]+}} <col:3, col:16> '<bound member function type>' ->~A
  a2->foo::A::~A();
  // CHECK: MemberExpr {{0x[0-9a-fA-F]+}} <col:3, col:13> '<bound member function type>' ->~P
  p1->~P<int>();
}

struct D {
  D(int);
  ~D();
};

void construct() {
  using namespace foo;
  A a = A(12);
  // CHECK: CXXConstructExpr {{0x[0-9a-fA-F]+}} <col:9, col:13> 'class foo::A' 'void (int){{( __attribute__\(\(thiscall\)\))?}}'
  D d = D(12);
  // CHECK: CXXConstructExpr {{0x[0-9a-fA-F]+}} <col:9, col:13> 'struct D' 'void (int){{( __attribute__\(\(thiscall\)\))?}}'
}
