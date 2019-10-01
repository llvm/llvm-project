// RUN: %clang_cc1 -verify -fcxx-exceptions -triple=x86_64-linux-gnu -std=c++11 -Werror=c++1y-extensions -Werror=c++2a-extensions %s
// RUN: %clang_cc1 -verify -fcxx-exceptions -triple=x86_64-linux-gnu -std=c++1y -DCXX1Y -Werror=c++2a-extensions %s
// RUN: %clang_cc1 -verify -fcxx-exceptions -triple=x86_64-linux-gnu -std=c++2a -DCXX1Y -DCXX2A %s

namespace N {
  typedef char C;
}

namespace M {
  typedef double D;
}

struct NonLiteral { // expected-note 3{{no constexpr constructors}}
  NonLiteral() {}
  NonLiteral(int) {}
};
struct Literal {
  constexpr Literal() {}
  operator int() const { return 0; }
};

struct S {
  virtual int ImplicitlyVirtual() const = 0;
#if __cplusplus <= 201703L
  // expected-note@-2 {{overridden virtual function}}
#endif
};
struct SS : S {
  int ImplicitlyVirtual() const;
};

// The definition of a constexpr function shall satisfy the following
// constraints:
struct T : SS, NonLiteral {
  constexpr T();
  constexpr int f() const;

  //  - it shall not be virtual; [until C++20]
  virtual constexpr int ExplicitlyVirtual() const { return 0; }
#if __cplusplus <= 201703L
  // expected-error@-2 {{virtual function cannot be constexpr}}
#endif

  constexpr int ImplicitlyVirtual() const { return 0; }
#if __cplusplus <= 201703L
  // expected-error@-2 {{virtual function cannot be constexpr}}
#endif

  virtual constexpr int OutOfLineVirtual() const;
#if __cplusplus <= 201703L
  // expected-error@-2 {{virtual function cannot be constexpr}}
#endif

  //  - its return type shall be a literal type;
  constexpr NonLiteral NonLiteralReturn() const { return {}; } // expected-error {{constexpr function's return type 'NonLiteral' is not a literal type}}
  constexpr void VoidReturn() const { return; }
#ifndef CXX1Y
  // expected-error@-2 {{constexpr function's return type 'void' is not a literal type}}
#endif
  constexpr ~T(); // expected-error {{destructor cannot be marked constexpr}}
  typedef NonLiteral F() const;
  constexpr F NonLiteralReturn2; // ok until definition

  //  - each of its parameter types shall be a literal type;
  constexpr int NonLiteralParam(NonLiteral) const { return 0; } // expected-error {{constexpr function's 1st parameter type 'NonLiteral' is not a literal type}}
  typedef int G(NonLiteral) const;
  constexpr G NonLiteralParam2; // ok until definition

  //  - its function-body shall be = delete, = default,
  constexpr int Deleted() const = delete;
  // It's not possible for the function-body to legally be "= default" here
  // (that is, for a non-constructor function) in C++11.
  // Other than constructors, only the copy- and move-assignment operators and
  // destructor can be defaulted. Destructors can't be constexpr since they
  // don't have a literal return type. Defaulted assignment operators can't be
  // constexpr since they can't be const.
  constexpr T &operator=(const T&) = default;
#ifndef CXX1Y
  // expected-error@-2 {{an explicitly-defaulted copy assignment operator may not have 'const', 'constexpr' or 'volatile' qualifiers}}
  // expected-warning@-3 {{C++14}}
#else
  // expected-error@-5 {{defaulted definition of copy assignment operator is not constexpr}}
#endif
};

constexpr int T::OutOfLineVirtual() const { return 0; }
#ifdef CXX1Y
struct T2 {
  int n = 0;
  constexpr T2 &operator=(const T2&) = default; // ok
};
struct T3 {
  constexpr T3 &operator=(const T3&) const = default;
#ifndef CXX2A
  // expected-error@-2 {{an explicitly-defaulted copy assignment operator may not have 'const' or 'volatile' qualifiers}}
#else
  // expected-warning@-4 {{explicitly defaulted copy assignment operator is implicitly deleted}}
  // expected-note@-5 {{function is implicitly deleted because its declared type does not match the type of an implicit copy assignment operator}}
#endif
};
#endif
struct U {
  constexpr U SelfReturn() const;
  constexpr int SelfParam(U) const;
};

struct V : virtual U { // expected-note {{here}}
  constexpr int F() const { return 0; } // expected-error {{constexpr member function not allowed in struct with virtual base class}}
};

//  or a compound-statememt that contains only [CXX11]
constexpr int AllowedStmtsCXX11() {
  //  - null statements
  ;

  //  - static_assert-declarations
  static_assert(true, "the impossible happened!");

  //  - typedef declarations and alias-declarations that do not define classes
  //    or enumerations
  typedef int I;
  typedef struct S T;
  using J = int;
  using K = int[sizeof(I) + sizeof(J)];
  // Note, the standard requires we reject this.
  struct U;

  //  - using-declarations
  using N::C;

  //  - using-directives
  using namespace N;

  //  - and exactly one return statement
  return sizeof(K) + sizeof(C) + sizeof(K);
}

//  or a compound-statement that does not contain [CXX1Y]
constexpr int DisallowedStmtsCXX1Y_1() {
  //  - an asm-definition
  asm("int3"); // expected-error {{statement not allowed in constexpr function}}
  return 0;
}
constexpr int DisallowedStmtsCXX1Y_2() {
  //  - a goto statement
  goto x; // expected-error {{statement not allowed in constexpr function}}
x:
  return 0;
}
constexpr int DisallowedStmtsCXX1Y_2_1() {
  try {
    return 0;
  } catch (...) {
  merp: goto merp; // expected-error {{statement not allowed in constexpr function}}
  }
}
constexpr int DisallowedStmtsCXX1Y_3() {
  //  - a try-block,
  try {} catch (...) {}
#ifndef CXX2A
  // expected-error@-2 {{use of this statement in a constexpr function is a C++2a extension}}
#ifndef CXX1Y
  // expected-error@-4 {{use of this statement in a constexpr function is a C++14 extension}}
#endif
#endif
  return 0;
}
constexpr int DisallowedStmtsCXX1Y_4() {
  //  - a definition of a variable of non-literal type
  NonLiteral nl; // expected-error {{variable of non-literal type 'NonLiteral' cannot be defined in a constexpr function}}
  return 0;
}
constexpr int DisallowedStmtsCXX1Y_5() {
  //  - a definition of a variable of static storage duration
  static constexpr int n = 123; // expected-error {{static variable not permitted in a constexpr function}}
  return n;
}
constexpr int DisallowedStmtsCXX1Y_6() {
  //  - a definition of a variable of thread storage duration
  thread_local constexpr int n = 123; // expected-error {{thread_local variable not permitted in a constexpr function}}
  return n;
}
constexpr int DisallowedStmtsCXX1Y_7() {
  //  - a definition of a variable for which no initialization is performed
  int n; // expected-error {{variables defined in a constexpr function must be initialized}}
  return 0;
}

constexpr int ForStmt() {
  for (int n = 0; n < 10; ++n)
#ifndef CXX1Y
  // expected-error@-2 {{statement not allowed in constexpr function}}
#endif
    return 0;
}
constexpr int VarDecl() {
  int a = 0;
#ifndef CXX1Y
  // expected-error@-2 {{variable declaration in a constexpr function is a C++14 extension}}
#endif
  return 0;
}
constexpr int ConstexprVarDecl() {
  constexpr int a = 0;
#ifndef CXX1Y
  // expected-error@-2 {{variable declaration in a constexpr function is a C++14 extension}}
#endif
  return 0;
}
constexpr int VarWithCtorDecl() {
  Literal a;
#ifndef CXX1Y
  // expected-error@-2 {{variable declaration in a constexpr function is a C++14 extension}}
#endif
  return 0;
}
NonLiteral nl;
constexpr NonLiteral &ExternNonLiteralVarDecl() {
  extern NonLiteral nl;
#ifndef CXX1Y
  // expected-error@-2 {{variable declaration in a constexpr function is a C++14 extension}}
#endif
  return nl;
}
static_assert(&ExternNonLiteralVarDecl() == &nl, "");
constexpr int FuncDecl() {
  constexpr int ForwardDecl(int);
#ifndef CXX1Y
  // expected-error@-2 {{use of this statement in a constexpr function is a C++14 extension}}
#endif
  return ForwardDecl(42);
}
constexpr int ClassDecl1() {
  typedef struct { } S1;
#ifndef CXX1Y
  // expected-error@-2 {{type definition in a constexpr function is a C++14 extension}}
#endif
  return 0;
}
constexpr int ClassDecl2() {
  using S2 = struct { };
#ifndef CXX1Y
  // expected-error@-2 {{type definition in a constexpr function is a C++14 extension}}
#endif
  return 0;
}
constexpr int ClassDecl3() {
  struct S3 { };
#ifndef CXX1Y
  // expected-error@-2 {{type definition in a constexpr function is a C++14 extension}}
#endif
  return 0;
}
constexpr int NoReturn() {} // expected-error {{no return statement in constexpr function}}
constexpr int MultiReturn() {
  return 0;
  return 0;
#ifndef CXX1Y
  // expected-error@-2 {{multiple return statements in constexpr function}}
  // expected-note@-4 {{return statement}}
#endif
}

//  - every constructor call and implicit conversion used in initializing the
//    return value shall be one of those allowed in a constant expression.
//
// We implement the proposed resolution of DR1364 and ignore this bullet.
// However, we implement the spirit of the check as part of the p5 checking that
// a constexpr function must be able to produce a constant expression.
namespace DR1364 {
  constexpr int f(int k) {
    return k; // ok, even though lvalue-to-rvalue conversion of a function
              // parameter is not allowed in a constant expression.
  }
  int kGlobal; // expected-note {{here}}
  constexpr int f() { // expected-error {{constexpr function never produces a constant expression}}
    return kGlobal; // expected-note {{read of non-const}}
  }
}

namespace rdar13584715 {
  typedef __PTRDIFF_TYPE__ ptrdiff_t;

  template<typename T> struct X {
    static T value() {};
  };

  void foo(ptrdiff_t id) {
    switch (id) {
    case reinterpret_cast<ptrdiff_t>(&X<long>::value):  // expected-error{{case value is not a constant expression}} \
      // expected-note{{reinterpret_cast is not allowed in a constant expression}}
      break;
    }
  }
}

namespace std_example {
  constexpr int square(int x) {
    return x * x;
  }
  constexpr long long_max() {
    return 2147483647;
  }
  constexpr int abs(int x) {
    if (x < 0)
#ifndef CXX1Y
      // expected-error@-2 {{C++14}}
#endif
      x = -x;
    return x;
  }
  constexpr int first(int n) {
    static int value = n; // expected-error {{static variable not permitted}}
    return value;
  }
  constexpr int uninit() {
    int a; // expected-error {{must be initialized}}
    return a;
  }
  constexpr int prev(int x) {
    return --x;
  }
#ifndef CXX1Y
  // expected-error@-4 {{never produces a constant expression}}
  // expected-note@-4 {{subexpression}}
#endif
  constexpr int g(int x, int n) {
    int r = 1;
    while (--n > 0) r *= x;
    return r;
  }
#ifndef CXX1Y
    // expected-error@-5 {{C++14}}
    // expected-error@-5 {{statement not allowed}}
#endif
}
