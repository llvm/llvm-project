// RUN: %check_clang_tidy %s misc-unused-using-decls %t -- --fix-notes -- -fno-delayed-template-parsing -isystem %S/Inputs

// ----- Definitions -----
template <typename T> class vector {};
namespace n {
class A;
class B;
class C;
class D;
class D { public: static int i; };
template <typename T> class E {};
template <typename T> class F {};
class G { public: static void func() {} };
class H { public: static int i; };
class I {
 public:
  static int ii;
};
template <typename T> class J {};
class G;
class H;

template <typename T> class K {};
template <template <typename> class S>
class L {};

template <typename T> class M {};
class N {};

template <int T> class P {};
const int Constant = 0;

template <typename T> class Q {};

class Base {
 public:
  void f();
};

D UsedInstance;
D UnusedInstance;

int UsedFunc() { return 1; }
int UnusedFunc() { return 1; }
template <typename T> int UsedTemplateFunc() { return 1; }
template <typename T> int UnusedTemplateFunc() { return 1; }
template <typename T> int UsedInTemplateFunc() { return 1; }
void OverloadFunc(int);
void OverloadFunc(double);
int FuncUsedByUsingDeclInMacro() { return 1; }
long double operator""_w(long double);

class ostream {
public:
  ostream &operator<<(ostream &(*PF)(ostream &));
};
extern ostream cout;
ostream &endl(ostream &os);

enum Color1 { Green };

enum Color2 { Red };

enum Color3 { Yellow };

enum Color4 { Blue };

}  // namespace n

#include "unused-using-decls.h"
namespace ns {
template <typename T>
class AA {
  T t;
};
template <typename T>
T ff() { T t; return t; }
} // namespace ns

// ----- Using declarations -----
// eol-comments aren't removed (yet)
using n::A; // A
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: using decl 'A' is unused
// CHECK-MESSAGES: :[[@LINE-2]]:10: note: remove the using
// CHECK-FIXES: {{^}}// A
using n::B;
using n::C;
using n::D;
using n::E; // E
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: using decl 'E' is unused
// CHECK-FIXES: {{^}}// E
using n::F;
using n::G;
using n::H;
using n::I;
int I::ii = 1;
class Derived : public n::Base {
 public:
  using Base::f;
};
using n::UsedInstance;
using n::UsedFunc;
using n::UsedTemplateFunc;
using n::UnusedInstance; // UnusedInstance
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: using decl 'UnusedInstance' is unused
// CHECK-FIXES: {{^}}// UnusedInstance
using n::UnusedFunc; // UnusedFunc
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: using decl 'UnusedFunc' is unused
// CHECK-FIXES: {{^}}// UnusedFunc
using n::operator""_w;
using n::cout;
using n::endl;

using n::UsedInTemplateFunc;
using n::J;
template <typename T> void Callee() {
  J<T> j;
  UsedInTemplateFunc<T>();
}

using n::OverloadFunc; // OverloadFunc
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: using decl 'OverloadFunc' is unused
// CHECK-FIXES: {{^}}// OverloadFunc

#define DEFINE_INT(name)        \
  namespace INT {               \
  static const int _##name = 1; \
  }                             \
  using INT::_##name
DEFINE_INT(test);
#undef DEFIND_INT

#define USING_FUNC \
  using n::FuncUsedByUsingDeclInMacro;
USING_FUNC
#undef USING_FUNC

namespace N1 {
// n::G is used in namespace N2.
// Currently, the check doesn't support multiple scopes. All the relevant
// using-decls will be marked as used once we see an usage even the usage is in
// other scope.
using n::G;
}

namespace N2 {
using n::G;
void f(G g);
}

void IgnoreFunctionScope() {
// Using-decls defined in function scope will be ignored.
using n::H;
}

using n::Color1;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: using decl 'Color1' is unused
using n::Green;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: using decl 'Green' is unused
using n::Color2;
using n::Color3;
using n::Blue;

using ns::AA;
using ns::ff;

using n::K;

using n::N;

// FIXME: Currently non-type template arguments are not supported.
using n::Constant;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: using decl 'Constant' is unused

using n::Q;

// ----- Usages -----
void f(B b);
void g() {
  vector<C> data;
  D::i = 1;
  F<int> f;
  void (*func)() = &G::func;
  int *i = &H::i;
  UsedInstance.i;
  UsedFunc();
  UsedTemplateFunc<int>();
  1.5_w;
  cout << endl;
  Color2 color2;
  int t1 = Color3::Yellow;
  int t2 = Blue;

  MyClass a;
  int t3 = 0;
  a.func1<AA>(&t3);
  a.func2<int, ff>(t3);

  n::L<K> l;
}

template<class T>
void h(n::M<T>* t) {}
// n::N is used the explicit template instantiation.
template void h(n::M<N>* t);

// Test on Non-type template arguments.
template <int T>
void i(n::P<T>* t) {}
template void i(n::P<Constant>* t);

template <typename T, template <typename> class U> class Bar {};
// We used to report Q unsued, because we only checked the first template
// argument.
Bar<int, Q> *bar;
