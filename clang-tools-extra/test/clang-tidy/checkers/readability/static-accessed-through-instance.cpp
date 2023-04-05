// RUN: %check_clang_tidy %s readability-static-accessed-through-instance %t -- -- -isystem %S/Inputs/static-accessed-through-instance
#include <__clang_cuda_builtin_vars.h>

enum OutEnum {
  E0,
};

struct C {
  static void foo();
  static int x;
  int nsx;
  enum {
    Anonymous,
  };
  enum E {
    E1,
  };
  using enum OutEnum;
  void mf() {
    (void)&x;    // OK, x is accessed inside the struct.
    (void)&C::x; // OK, x is accessed using a qualified-id.
    foo();       // OK, foo() is accessed inside the struct.
  }
  void ns() const;
};

int C::x = 0;

struct CC {
  void foo();
  int x;
};

template <typename T> struct CT {
  static T foo();
  static T x;
  int nsx;
  void mf() {
    (void)&x;    // OK, x is accessed inside the struct.
    (void)&C::x; // OK, x is accessed using a qualified-id.
    foo();       // OK, foo() is accessed inside the struct.
  }
};

// Expressions with side effects
C &f(int, int, int, int);
void g() {
  f(1, 2, 3, 4).x;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: static member accessed through instance  [readability-static-accessed-through-instance]
  // CHECK-FIXES: {{^}}  f(1, 2, 3, 4).x;{{$}}
}

int i(int &);
void j(int);
C h();
bool a();
int k(bool);

void f(C c) {
  j(i(h().x));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: static member
  // CHECK-FIXES: {{^}}  j(i(h().x));{{$}}

  // The execution of h() depends on the return value of a().
  j(k(a() && h().x));
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: static member
  // CHECK-FIXES: {{^}}  j(k(a() && h().x));{{$}}

  if ([c]() {
        c.ns();
        return c;
      }().x == 15)
    ;
  // CHECK-MESSAGES: :[[@LINE-5]]:7: warning: static member
  // CHECK-FIXES: {{^}}  if ([c]() {{{$}}
}

// Nested specifiers
namespace N {
struct V {
  static int v;
  struct T {
    static int t;
    struct U {
      static int u;
    };
  };
};
}

void f(N::V::T::U u) {
  N::V v;
  v.v = 12;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: static member
  // CHECK-FIXES: {{^}}  N::V::v = 12;{{$}}

  N::V::T w;
  w.t = 12;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: static member
  // CHECK-FIXES: {{^}}  N::V::T::t = 12;{{$}}

  // u.u is not changed to N::V::T::U::u; because the nesting level is over 3.
  u.u = 12;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: static member
  // CHECK-FIXES: {{^}}  u.u = 12;{{$}}

  using B = N::V::T::U;
  B b;
  b.u;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: static member
  // CHECK-FIXES: {{^}}  B::u;{{$}}
}

// Templates
template <typename T> T CT<T>::x;

template <typename T> struct CCT {
  T foo();
  T x;
};

typedef C D;

using E = D;

#define FOO(c) c.foo()
#define X(c) c.x

template <typename T> void f(T t, C c) {
  t.x; // OK, t is a template parameter.
  c.x;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: static member
  // CHECK-FIXES: {{^}}  C::x;{{$}}
}

template <int N> struct S { static int x; };

template <> struct S<0> { int x; };

template <int N> void h() {
  S<N> sN;
  sN.x; // OK, value of N affects whether x is static or not.

  S<2> s2;
  s2.x;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: static member
  // CHECK-FIXES: {{^}}  S<2>::x;{{$}}
}

void static_through_instance() {
  C *c1 = new C();
  c1->foo(); // 1
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: static member
  // CHECK-FIXES: {{^}}  C::foo(); // 1{{$}}
  c1->x; // 2
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: static member
  // CHECK-FIXES: {{^}}  C::x; // 2{{$}}
  c1->Anonymous; // 3
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: static member
  // CHECK-FIXES: {{^}}  C::Anonymous; // 3{{$}}
  c1->E1; // 4
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: static member
  // CHECK-FIXES: {{^}}  C::E1; // 4{{$}}
  c1->E0; // 5
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: static member
  // CHECK-FIXES: {{^}}  C::E0; // 5{{$}}

  c1->nsx; // OK, nsx is a non-static member.

  const C *c2 = new C();
  c2->foo(); // 2
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: static member
  // CHECK-FIXES: {{^}}  C::foo(); // 2{{$}}

  C::foo(); // OK, foo() is accessed using a qualified-id.
  C::x;     // OK, x is accessed using a qualified-id.

  D d;
  d.foo();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: static member
  // CHECK-FIXES: {{^}}  D::foo();{{$}}
  d.x;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: static member
  // CHECK-FIXES: {{^}}  D::x;{{$}}

  E e;
  e.foo();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: static member
  // CHECK-FIXES: {{^}}  E::foo();{{$}}
  e.x;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: static member
  // CHECK-FIXES: {{^}}  E::x;{{$}}

  CC *cc = new CC;

  f(*c1, *c1);
  f(*cc, *c1);

  // Macros: OK, macros are not checked.
  FOO((*c1));
  X((*c1));
  FOO((*cc));
  X((*cc));

  // Templates
  CT<int> ct;
  ct.foo();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: static member
  // CHECK-FIXES: {{^}}  CT<int>::foo();{{$}}
  ct.x;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: static member
  // CHECK-FIXES: {{^}}  CT<int>::x;{{$}}
  ct.nsx; // OK, nsx is a non-static member

  CCT<int> cct;
  cct.foo(); // OK, CCT has no static members.
  cct.x;     // OK, CCT has no static members.

  h<4>();
}

struct SP {
  static int I;
} P;

void usep() {
  P.I;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: static member
  // CHECK-FIXES: {{^}}  SP::I;{{$}}
}

namespace NSP {
struct SP {
  static int I;
} P;
} // namespace NSP

void usensp() {
  NSP::P.I;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: static member
  // CHECK-FIXES: {{^}}  NSP::SP::I;{{$}}
}

// Overloaded member access operator
struct Q {
  static int K;
  int y = 0;
};

int Q::K = 0;

struct Qptr {
  Q *q;

  explicit Qptr(Q *qq) : q(qq) {}

  Q *operator->() {
    ++q->y;
    return q;
  }
};

int func(Qptr qp) {
  qp->y = 10; // OK, the overloaded operator might have side-effects.
  qp->K = 10; //
}

namespace {
  struct Anonymous {
    static int I;
  };
}

void use_anonymous() {
  Anonymous Anon;
  Anon.I;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: static member
  // CHECK-FIXES: {{^}}  Anonymous::I;{{$}}
}

namespace Outer {
  inline namespace Inline {
  struct S {
    static int I;
  };
  }
}

void use_inline() {
  Outer::S V;
  V.I;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: static member
  // CHECK-FIXES: {{^}}  Outer::S::I;{{$}}
}

// https://bugs.llvm.org/show_bug.cgi?id=48758
namespace Bugzilla_48758 {

unsigned int x1 = threadIdx.x;
// CHECK-MESSAGES-NOT: :[[@LINE-1]]:10: warning: static member
unsigned int x2 = blockIdx.x;
// CHECK-MESSAGES-NOT: :[[@LINE-1]]:10: warning: static member
unsigned int x3 = blockDim.x;
// CHECK-MESSAGES-NOT: :[[@LINE-1]]:10: warning: static member
unsigned int x4 = gridDim.x;
// CHECK-MESSAGES-NOT: :[[@LINE-1]]:10: warning: static member

} // namespace Bugzilla_48758

// https://github.com/llvm/llvm-project/issues/61736
namespace llvm_issue_61736
{

struct {
  static void f() {}
} AnonStruct, *AnonStructPointer;

class {
  public:
  static void f() {}
} AnonClass, *AnonClassPointer;

void testAnonymousStructAndClass() {
  AnonStruct.f();
  AnonStructPointer->f();

  AnonClass.f();
  AnonClassPointer->f();
}

struct Embedded {
  struct {
    static void f() {}
  } static EmbeddedStruct, *EmbeddedStructPointer;

  class {
    public:
      static void f() {}
  } static EmbeddedClass, *EmbeddedClassPointer;
};

void testEmbeddedAnonymousStructAndClass() {
  Embedded::EmbeddedStruct.f();
  Embedded::EmbeddedStructPointer->f();

  Embedded::EmbeddedClass.f();
  Embedded::EmbeddedClassPointer->f();

  Embedded E;
  E.EmbeddedStruct.f();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: static member accessed through instance [readability-static-accessed-through-instance]
  // CHECK-FIXES: {{^}}  llvm_issue_61736::Embedded::EmbeddedStruct.f();{{$}}
  E.EmbeddedStructPointer->f();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: static member accessed through instance [readability-static-accessed-through-instance]
  // CHECK-FIXES: {{^}}  llvm_issue_61736::Embedded::EmbeddedStructPointer->f();{{$}}

  E.EmbeddedClass.f();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: static member accessed through instance [readability-static-accessed-through-instance]
  // CHECK-FIXES: {{^}}  llvm_issue_61736::Embedded::EmbeddedClass.f();{{$}}
  E.EmbeddedClassPointer->f();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: static member accessed through instance [readability-static-accessed-through-instance]
  // CHECK-FIXES: {{^}}  llvm_issue_61736::Embedded::EmbeddedClassPointer->f();{{$}}
}

} // namespace llvm_issue_61736
