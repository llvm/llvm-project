// RUN: %clang_cc1 -verify -std=c++11 %s

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
  virtual int ImplicitlyVirtual() const = 0; // expected-note {{overridden virtual function}}
};
struct SS : S { 
  int ImplicitlyVirtual() const;
};

// The definition of a constexpr function shall satisfy the following
// constraints:
struct T : SS, NonLiteral { // expected-note {{base class 'NonLiteral' of non-literal type}}
  constexpr T();
  constexpr int f(); // expected-error {{non-literal type 'T' cannot have constexpr members}}

  //  - it shall not be virtual;
  virtual constexpr int ExplicitlyVirtual() { return 0; } // expected-error {{virtual function cannot be constexpr}}

  constexpr int ImplicitlyVirtual() { return 0; } // expected-error {{virtual function cannot be constexpr}}

  //  - its return type shall be a literal type;
  constexpr NonLiteral NonLiteralReturn() { return {}; } // expected-error {{constexpr function's return type 'NonLiteral' is not a literal type}}
  constexpr void VoidReturn() { return; } // expected-error {{constexpr function's return type 'void' is not a literal type}}
  constexpr ~T(); // expected-error {{destructor cannot be marked constexpr}}
  typedef NonLiteral F();
  constexpr F NonLiteralReturn2; // ok until definition

  //  - each of its parameter types shall be a literal type;
  constexpr int NonLiteralParam(NonLiteral) { return 0; } // expected-error {{constexpr function's 1st parameter type 'NonLiteral' is not a literal type}}
  typedef int G(NonLiteral);
  constexpr G NonLiteralParam2; // ok until definition

  //  - its function-body shall be = delete, = default,
  constexpr int Deleted() = delete;
  // It's not possible for the function-body to legally be "= default" here.
  // Other than constructors, only the copy- and move-assignment operators and
  // destructor can be defaulted. Destructors can't be constexpr since they
  // don't have a literal return type. Defaulted assignment operators can't be
  // constexpr since they can't be const.
  constexpr T &operator=(const T&) = default; // expected-error {{an explicitly-defaulted copy assignment operator may not have 'const', 'constexpr' or 'volatile' qualifiers}}
};
struct U {
  constexpr U SelfReturn();
  constexpr int SelfParam(U);
};

struct V : virtual U { // expected-note {{here}}
  constexpr int F() { return 0; } // expected-error {{constexpr member function not allowed in struct with virtual base class}}
};

//  or a compound-statememt that contains only
constexpr int AllowedStmts() {
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
constexpr int ForStmt() {
  for (int n = 0; n < 10; ++n) // expected-error {{statement not allowed in constexpr function}}
    return 0;
}
constexpr int VarDecl() {
  constexpr int a = 0; // expected-error {{variables cannot be declared in a constexpr function}}
  return 0;
}
constexpr int FuncDecl() {
  constexpr int ForwardDecl(int); // expected-error {{statement not allowed in constexpr function}}
  return ForwardDecl(42);
}
constexpr int ClassDecl1() {
  typedef struct { } S1; // expected-error {{types cannot be defined in a constexpr function}}
  return 0;
}
constexpr int ClassDecl2() {
  using S2 = struct { }; // expected-error {{types cannot be defined in a constexpr function}}
  return 0;
}
constexpr int ClassDecl3() {
  struct S3 { }; // expected-error {{types cannot be defined in a constexpr function}}
  return 0;
}
constexpr int NoReturn() {} // expected-error {{no return statement in constexpr function}}
constexpr int MultiReturn() {
  return 0; // expected-note {{return statement}}
  return 0; // expected-error {{multiple return statements in constexpr function}}
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
