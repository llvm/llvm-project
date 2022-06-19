// RUN: %clang_cc1 -fcxx-exceptions -verify=expected,beforecxx14,beforecxx20,beforecxx2b -std=c++11 %s
// RUN: %clang_cc1 -fcxx-exceptions -verify=expected,aftercxx14,beforecxx20,beforecxx2b -std=c++14 %s
// RUN: %clang_cc1 -fcxx-exceptions -verify=expected,aftercxx14,aftercxx20,beforecxx2b -std=c++20  %s
// RUN: %clang_cc1 -fcxx-exceptions -verify=expected,aftercxx14,aftercxx20 -std=c++2b %s

namespace N {
  typedef char C;
}

namespace M {
  typedef double D;
}

struct NonLiteral { // expected-note 2{{no constexpr constructors}}
  NonLiteral() {}
  NonLiteral(int) {}
};
struct Literal {
  constexpr Literal() {}
  operator int() const { return 0; }
};

struct S {
  virtual int ImplicitlyVirtual() const = 0; // beforecxx20-note {{overridden virtual function}}
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
  virtual constexpr int ExplicitlyVirtual() const { return 0; } // beforecxx20-error {{virtual function cannot be constexpr}}

  constexpr int ImplicitlyVirtual() const { return 0; } // beforecxx20-error {{virtual function cannot be constexpr}}

  virtual constexpr int OutOfLineVirtual() const; // beforecxx20-error {{virtual function cannot be constexpr}}

  //  - its return type shall be a literal type;
  constexpr NonLiteral NonLiteralReturn() const { return {}; } // expected-error {{constexpr function's return type 'NonLiteral' is not a literal type}}
  constexpr void VoidReturn() const { return; }                // beforecxx14-error {{constexpr function's return type 'void' is not a literal type}}
  constexpr ~T();                                              // beforecxx20-error {{destructor cannot be declared constexpr}}

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
  constexpr T &operator=(const T &) = default; // beforecxx14-error {{an explicitly-defaulted copy assignment operator may not have 'const', 'constexpr' or 'volatile' qualifiers}} \
                                               // beforecxx14-warning {{C++14}} \
                                               // aftercxx14-error{{defaulted definition of copy assignment operator is not constexpr}}
};

constexpr int T::OutOfLineVirtual() const { return 0; }
#if __cplusplus >= 201402L
struct T2 {
  int n = 0;
  constexpr T2 &operator=(const T2&) = default; // ok
};
struct T3 {
  constexpr T3 &operator=(const T3 &) const = default; // beforecxx20-error {{an explicitly-defaulted copy assignment operator may not have 'const' or 'volatile' qualifiers}} \
                                                       // aftercxx20-warning {{explicitly defaulted copy assignment operator is implicitly deleted}} \
                                                       // aftercxx20-note {{function is implicitly deleted because its declared type does not match the type of an implicit copy assignment operator}}
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

//  or a compound-statement that does not contain [C++14]
constexpr int DisallowedStmtsCXX14_1(bool b) {
  //  - an asm-definition
  if (b)
    asm("int3"); // beforecxx20-warning {{use of this statement in a constexpr function is a C++20 extension}}
  return 0;
}
constexpr int DisallowedStmtsCXX14_2() {
  return 0; // beforecxx14-note {{previous}}
  //  - a goto statement
  goto x; // beforecxx2b-warning {{use of this statement in a constexpr function is a C++2b extension}}
  x:;
    return 0; // beforecxx14-warning {{multiple return}}
}
constexpr int DisallowedStmtsCXX14_2_1() {
merp: // beforecxx2b-warning {{use of this statement in a constexpr function is a C++2b extension}}
  return 0;
}
constexpr int DisallowedStmtsCXX14_3() {
  //  - a try-block,
  try {  }  // beforecxx20-warning {{use of this statement in a constexpr function is a C++20 extension}}
  catch (...) {}
  return 0;
}
constexpr int DisallowedStmtsCXX14_4() {
  //  - a definition of a variable of non-literal type
  return 0;
  NonLiteral nl; // beforecxx2b-error {{variable of non-literal type 'NonLiteral' cannot be defined in a constexpr function before C++2b}} \
                 // beforecxx2b-note@14  {{'NonLiteral' is not literal}}
}

constexpr int DisallowedStmtsCXX14_5() {
  return 0;
  //  - a definition of a variable of static storage duration
  static constexpr int n = 123; // beforecxx2b-warning {{definition of a static variable in a constexpr function is a C++2b extension}} \
                               //  beforecxx14-warning {{variable declaration in a constexpr function is a C++14 extension}}
}

constexpr int DisallowedStmtsCXX14_6() {
  //  - a definition of a variable of thread storage duration
  return 0;
  thread_local constexpr int n = 123; // beforecxx14-warning {{variable declaration in a constexpr function is a C++14 extension}} \
                                      // beforecxx2b-warning {{definition of a thread_local variable in a constexpr function is a C++2b extension}}
}
constexpr int DisallowedStmtsCXX14_7() {
  //  - a definition of a variable for which no initialization is performed
  return 0;
  int n; // beforecxx20-warning {{uninitialized variable in a constexpr function}}
}

constexpr int ForStmt() {
  for (int n = 0; n < 10; ++n) {} // beforecxx14-error {{statement not allowed in constexpr function}}
    return 0;
}

constexpr int VarDecl() {
  int a = 0; // beforecxx14-warning {{variable declaration in a constexpr function is a C++14 extension}}
  return 0;
}
constexpr int ConstexprVarDecl() {
  constexpr int a = 0; // beforecxx14-warning {{variable declaration in a constexpr function is a C++14 extension}}
  return 0;
}
constexpr int VarWithCtorDecl() {
  Literal a; // beforecxx14-warning {{variable declaration in a constexpr function is a C++14 extension}}
  return 0;
}

NonLiteral nl;
constexpr NonLiteral &ExternNonLiteralVarDecl() {
  extern NonLiteral nl; // beforecxx14-warning {{variable declaration in a constexpr function is a C++14 extension}}
  return nl;
}
static_assert(&ExternNonLiteralVarDecl() == &nl, "");

constexpr int FuncDecl() {
  constexpr int ForwardDecl(int); // beforecxx14-warning {{use of this statement in a constexpr function is a C++14 extension}}
  return ForwardDecl(42);
}

constexpr int ClassDecl1() {
  typedef struct {} S1; // beforecxx14-warning {{type definition in a constexpr function is a C++14 extension}}
  return 0;
}

constexpr int ClassDecl2() {
  using S2 = struct {}; // beforecxx14-warning {{type definition in a constexpr function is a C++14 extension}}
  return 0;
}

constexpr int ClassDecl3() {
  struct S3 {}; // beforecxx14-warning {{type definition in a constexpr function is a C++14 extension}}
  return 0;
}

constexpr int NoReturn() {} // expected-error {{no return statement in constexpr function}}
constexpr int MultiReturn() {
  return 0; // beforecxx14-note {{return statement}}
  return 0; // beforecxx14-warning {{multiple return statements in constexpr function}}
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
    return kGlobal;   // expected-note {{read of non-const}}
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
    if (x < 0) // beforecxx14-warning {{C++14}}
      x = -x;
    return x;
  }
  constexpr int first(int n) {
    return 0;
    static int value = n; // beforecxx2b-warning {{definition of a static variable in a constexpr function is a C++2b extension}} \
                          // beforecxx14-warning {{variable declaration in a constexpr function is a C++14 extension}}
  }
  constexpr int uninit() {
    int a; // beforecxx20-warning {{uninitialized}}
    return a;
  }
  constexpr int prev(int x) { // beforecxx14-error {{never produces a constant expression}}
    return --x;               // beforecxx14-note {{subexpression}}
  }

  constexpr int g(int x, int n) {
    int r = 1; // beforecxx14-warning{{C++14}}
    while (--n > 0) // beforecxx14-error {{statement not allowed in constexpr function}}
      r *= x;
    return r;
  }
}
