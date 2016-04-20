// RUN: %check_clang_tidy %s misc-unused-using-decls %t

// ----- Definitions -----
template <typename T> class vector {};
namespace n {
class A;
class B;
class C;
class D;
class D { public: static int i; };
}

// ----- Using declarations -----
// eol-comments aren't removed (yet)
using n::A; // A
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: using decl 'A' is unused
// CHECK-FIXES: {{^}}// A
using n::B;
using n::C;
using n::D;

// ----- Usages -----
void f(B b);
void g() {
  vector<C> data;
  D::i = 1;
}

