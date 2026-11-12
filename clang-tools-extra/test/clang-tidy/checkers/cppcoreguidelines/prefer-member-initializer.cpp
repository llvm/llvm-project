// RUN: %check_clang_tidy %s cppcoreguidelines-prefer-member-initializer %t -- -- -fcxx-exceptions

extern void __assert_fail (__const char *__assertion, __const char *__file,
    unsigned int __line, __const char *__function)
     __attribute__ ((__noreturn__));
#define assert(expr) \
  ((expr)  ? (void)(0)  : __assert_fail (#expr, __FILE__, __LINE__, __func__))

class Simple1 {
  int n;
  double x;

public:
  Simple1() {
    // CHECK-FIXES: Simple1() : n(0), x(0.0) {
    n = 0;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'n' should be initialized in a member initializer of the constructor [cppcoreguidelines-prefer-member-initializer]
    // CHECK-FIXES: {{^\ *$}}
    x = 0.0;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'x' should be initialized in a member initializer of the constructor [cppcoreguidelines-prefer-member-initializer]
    // CHECK-FIXES: {{^\ *$}}
  }

  Simple1(int nn, double xx) {
    // CHECK-FIXES: Simple1(int nn, double xx) : n(nn), x(xx) {
    n = nn;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'n' should be initialized in a member initializer of the constructor [cppcoreguidelines-prefer-member-initializer]
    // CHECK-FIXES: {{^\ *$}}
    x = xx;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'x' should be initialized in a member initializer of the constructor [cppcoreguidelines-prefer-member-initializer]
    // CHECK-FIXES: {{^\ *$}}
  }

  ~Simple1() = default;
};

class Simple2 {
  int n;
  double x;

public:
  Simple2() : n(0) {
    // CHECK-FIXES: Simple2() : n(0), x(0.0) {
    x = 0.0;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'x' should be initialized in a member initializer of the constructor [cppcoreguidelines-prefer-member-initializer]
    // CHECK-FIXES: {{^\ *$}}
  }

  Simple2(int nn, double xx) : n(nn) {
    // CHECK-FIXES: Simple2(int nn, double xx) : n(nn), x(xx) {
    x = xx;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'x' should be initialized in a member initializer of the constructor [cppcoreguidelines-prefer-member-initializer]
    // CHECK-FIXES: {{^\ *$}}
  }

  ~Simple2() = default;
};

class Simple3 {
  int n;
  double x;

public:
  Simple3() : x(0.0) {
    // CHECK-FIXES: Simple3() : n(0), x(0.0) {
    n = 0;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'n' should be initialized in a member initializer of the constructor [cppcoreguidelines-prefer-member-initializer]
    // CHECK-FIXES: {{^\ *$}}
  }

  Simple3(int nn, double xx) : x(xx) {
    // CHECK-FIXES: Simple3(int nn, double xx) : n(nn), x(xx) {
    n = nn;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'n' should be initialized in a member initializer of the constructor [cppcoreguidelines-prefer-member-initializer]
    // CHECK-FIXES: {{^\ *$}}
  }

  ~Simple3() = default;
};

int something_int();
double something_double();

class Simple4 {
  int n;

public:
  Simple4() {
    // CHECK-FIXES: Simple4() : n(something_int()) {
    n = something_int();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'n' should be initialized in a member initializer of the constructor [cppcoreguidelines-prefer-member-initializer]
    // CHECK-FIXES: {{^\ *$}}
  }

  ~Simple4() = default;
};

static bool dice();

class Complex1 {
  int n;
  int m;

public:
  Complex1() : n(0) {
    if (dice())
      m = 1;
    // NO-MESSAGES: initialization of 'm' is nested in a conditional expression
  }

  ~Complex1() = default;
};

class Complex2 {
  int n;
  int m;

public:
  Complex2() : n(0) {
    if (!dice())
      return;
    m = 1;
    // NO-MESSAGES: initialization of 'm' follows a conditional expression
  }

  ~Complex2() = default;
};

class Complex3 {
  int n;
  int m;

public:
  Complex3() : n(0) {
    while (dice())
      m = 1;
    // NO-MESSAGES: initialization of 'm' is nested in a conditional loop
  }

  ~Complex3() = default;
};

class Complex4 {
  int n;
  int m;

public:
  Complex4() : n(0) {
    while (!dice())
      return;
    m = 1;
    // NO-MESSAGES: initialization of 'm' follows a conditional loop
  }

  ~Complex4() = default;
};

class Complex5 {
  int n;
  int m;

public:
  Complex5() : n(0) {
    do {
      m = 1;
      // NO-MESSAGES: initialization of 'm' is nested in a conditional loop
    } while (dice());
  }

  ~Complex5() = default;
};

class Complex6 {
  int n;
  int m;

public:
  Complex6() : n(0) {
    do {
      return;
    } while (!dice());
    m = 1;
    // NO-MESSAGES: initialization of 'm' follows a conditional loop
  }

  ~Complex6() = default;
};

class Complex7 {
  int n;
  int m;

public:
  Complex7() : n(0) {
    for (int i = 2; i < 1; ++i) {
      m = 1;
    }
    // NO-MESSAGES: initialization of 'm' is nested into a conditional loop
  }

  ~Complex7() = default;
};

class Complex8 {
  int n;
  int m;

public:
  Complex8() : n(0) {
    for (int i = 0; i < 2; ++i) {
      return;
    }
    m = 1;
    // NO-MESSAGES: initialization of 'm' follows a conditional loop
  }

  ~Complex8() = default;
};

class Complex9 {
  int n;
  int m;

public:
  Complex9() : n(0) {
    switch (dice()) {
    case 1:
      m = 1;
      // NO-MESSAGES: initialization of 'm' is nested in a conditional expression
      break;
    default:
      break;
    }
  }

  ~Complex9() = default;
};

class Complex10 {
  int n;
  int m;

public:
  Complex10() : n(0) {
    switch (dice()) {
    case 1:
      return;
      break;
    default:
      break;
    }
    m = 1;
    // NO-MESSAGES: initialization of 'm' follows a conditional expression
  }

  ~Complex10() = default;
};

class E {};
int risky(); // may throw

class Complex11 {
  int n;
  int m;

public:
  Complex11() : n(0) {
    try {
      risky();
      m = 1;
      // NO-MESSAGES: initialization of 'm' follows is nested in a try-block
    } catch (const E& e) {
      return;
    }
  }

  ~Complex11() = default;
};

class Complex12 {
  int n;
  int m;

public:
  Complex12() : n(0) {
    try {
      risky();
    } catch (const E& e) {
      return;
    }
    m = 1;
    // NO-MESSAGES: initialization of 'm' follows a try-block
  }

  ~Complex12() = default;
};

class Complex13 {
  int n;
  int m;

public:
  Complex13() : n(0) {
    return;
    m = 1;
    // NO-MESSAGES: initialization of 'm' follows a return statement
  }

  ~Complex13() = default;
};

class Complex14 {
  int n;
  int m;

public:
  Complex14() : n(0) {
    goto X;
    m = 1;
    // NO-MESSAGES: initialization of 'm' follows a goto statement
  X:
    ;
  }

  ~Complex14() = default;
};

void returning();

class Complex15 {
  int n;
  int m;

public:
  Complex15() : n(0) {
    // CHECK-FIXES: Complex15() : n(0), m(1) {
    returning();
    m = 1;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'm' should be initialized in a member initializer of the constructor [cppcoreguidelines-prefer-member-initializer]
    // CHECK-FIXES: {{^\ *$}}
  }

  ~Complex15() = default;
};

[[noreturn]] void not_returning();

class Complex16 {
  int n;
  int m;

public:
  Complex16() : n(0) {
    not_returning();
    m = 1;
    // NO-MESSAGES: initialization of 'm' follows a non-returning function call
  }

  ~Complex16() = default;
};

class Complex17 {
  int n;
  int m;

public:
  Complex17() : n(0) {
    throw 1;
    m = 1;
    // NO-MESSAGES: initialization of 'm' follows a 'throw' statement;
  }

  ~Complex17() = default;
};

class Complex18 {
  int n;

public:
  Complex18() try {
    n = risky();
    // NO-MESSAGES: initialization of 'n' in a 'try' body;
  } catch (const E& e) {
    n = 0;
  }

  ~Complex18() = default;
};

class Complex19 {
  int n;
public:
  Complex19() {
    // CHECK-FIXES: Complex19() : n(0) {
    n = 0;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'n' should be initialized in a member initializer of the constructor [cppcoreguidelines-prefer-member-initializer]
    // CHECK-FIXES: {{^\ *$}}
  }

  explicit Complex19(int) {
    // CHECK-FIXES: explicit Complex19(int) : n(12) {
    n = 12;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'n' should be initialized in a member initializer of the constructor [cppcoreguidelines-prefer-member-initializer]
    // CHECK-FIXES: {{^\ *$}}
  }

  ~Complex19() = default;
};

class Complex20 {
  int n;
  int m;

public:
  Complex20(int k) : n(0) {
    assert(k > 0);
    m = 1;
    // NO-MESSAGES: initialization of 'm' follows an assertion
  }

  ~Complex20() = default;
};

class VeryComplex1 {
  int n1, n2, n3;
  double x1, x2, x3;
  int n4, n5, n6;
  double x4, x5, x6;

  VeryComplex1() : n3(something_int()), x3(something_double()),
                   n5(something_int()), x4(something_double()),
                   x5(something_double()) {
    // CHECK-FIXES: VeryComplex1() : n2(something_int()), n1(something_int()), n3(something_int()), x2(something_double()), x1(something_double()), x3(something_double()),
    // CHECK-FIXES:                  n4(something_int()), n5(something_int()), n6(something_int()), x4(something_double()),
    // CHECK-FIXES:                  x5(something_double()), x6(something_double()) {

// FIXME: Order of elements on the constructor initializer list should match
//        the order of the declaration of the fields. Thus the correct fixes
//        should look like these:
//
    // C ECK-FIXES: VeryComplex1() : n2(something_int()), n1(something_int()), n3(something_int()), x2(something_double()), x1(something_double()), x3(something_double()),
    // C ECK-FIXES:                  n4(something_int()), n5(something_int()), n6(something_int()), x4(something_double()),
    // C ECK-FIXES:                  x5(something_double()), x6(something_double()) {
//
//        However, the Diagnostics Engine processes fixes in the order of the
//        diagnostics and insertions to the same position are handled in left to
//        right order thus in the case two adjacent fields are initialized
//        inside the constructor in reverse order the provided fix is a
//        constructor initializer list that does not match the order of the
//        declaration of the fields.

    x2 = something_double();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'x2' should be initialized in a member initializer of the constructor [cppcoreguidelines-prefer-member-initializer]
    // CHECK-FIXES: {{^\ *$}}
    n2 = something_int();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'n2' should be initialized in a member initializer of the constructor [cppcoreguidelines-prefer-member-initializer]
    // CHECK-FIXES: {{^\ *$}}
    x6 = something_double();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'x6' should be initialized in a member initializer of the constructor [cppcoreguidelines-prefer-member-initializer]
    // CHECK-FIXES: {{^\ *$}}
    x1 = something_double();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'x1' should be initialized in a member initializer of the constructor [cppcoreguidelines-prefer-member-initializer]
    // CHECK-FIXES: {{^\ *$}}
    n6 = something_int();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'n6' should be initialized in a member initializer of the constructor [cppcoreguidelines-prefer-member-initializer]
    // CHECK-FIXES: {{^\ *$}}
    n1 = something_int();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'n1' should be initialized in a member initializer of the constructor [cppcoreguidelines-prefer-member-initializer]
    // CHECK-FIXES: {{^\ *$}}
    n4 = something_int();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'n4' should be initialized in a member initializer of the constructor [cppcoreguidelines-prefer-member-initializer]
    // CHECK-FIXES: {{^\ *$}}
  }
};

struct Outside {
  int n;
  double x;
  Outside();
};

Outside::Outside() {
    // CHECK-FIXES: Outside::Outside() : n(1), x(1.0) {
  n = 1;
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'n' should be initialized in a member initializer of the constructor [cppcoreguidelines-prefer-member-initializer]
    // CHECK-FIXES: {{^\ *$}}
  x = 1.0;
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'x' should be initialized in a member initializer of the constructor [cppcoreguidelines-prefer-member-initializer]
    // CHECK-FIXES: {{^\ *$}}
}

struct SafeDependancy {
  int m;
  int n;
  SafeDependancy(int M) : m(M) {
    // CHECK-FIXES: SafeDependancy(int M) : m(M), n(m) {
    n = m;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'n' should be initialized in a member initializer of the constructor
  }
  // We match against direct field dependancy as well as descendant field
  // dependancy, ensure both are accounted for.
  SafeDependancy(short M) : m(M) {
    // CHECK-FIXES: SafeDependancy(short M) : m(M), n(m + 1) {
    n = m + 1;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'n' should be initialized in a member initializer of the constructor
  }
};

struct BadDependancy {
  int m;
  int n;
  BadDependancy(int N) : n(N) {
    m = n;
  }
  BadDependancy(short N) : n(N) {
    m = n + 1;
  }
};

struct InitFromVarDecl {
  int m;
  InitFromVarDecl() {
    // Can't apply this fix as n is declared in the body of the constructor.
    int n = 3;
    m = n;
  }
};

struct HasInClassInit {
  int m = 4;
  HasInClassInit() {
    m = 3;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'm' should be initialized in a member initializer of the constructor
  }
};

struct HasInitListInit {
  int M;
  // CHECK-MESSAGES: :[[@LINE+5]]:5: warning: 'M' should be initialized in a member initializer of the constructor
  // CHECK-FIXES: HasInitListInit(const HasInitListInit &Other) : M(Other.M) {
  // CHECK-FIXES-NEXT: {{^    $}}
  // CHECK-FIXES-NEXT: }
  HasInitListInit(const HasInitListInit &Other) : M(4) {
    M = Other.M;
  }
  // CHECK-MESSAGES: :[[@LINE+5]]:5: warning: 'M' should be initialized in a member initializer of the constructor
  // CHECK-FIXES: HasInitListInit(HasInitListInit &&Other) : M(Other.M) {
  // CHECK-FIXES-NEXT: {{^    $}}
  // CHECK-FIXES-NEXT: }
  HasInitListInit(HasInitListInit &&Other) : M() {
    M = Other.M;
  }
};

#define ASSIGN_IN_MACRO(FIELD, VALUE) FIELD = (VALUE);

struct MacroCantFix {
  int n; // NoFix
  // CHECK-FIXES: int n; // NoFix
  MacroCantFix() {
    ASSIGN_IN_MACRO(n, 0)
    // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: 'n' should be initialized in a member initializer of the constructor
    // CHECK-FIXES: ASSIGN_IN_MACRO(n, 0)
  }
};

struct PR52818  {
    PR52818() : bar(5) {}
    PR52818(int) : PR52818() { bar = 3; }

    int bar;
};

struct RefReassignment {
  RefReassignment(int &i) : m_i{i} {
    m_i = 1;
  }
  int & m_i;
};

struct ReassignmentAfterUnsafetyAssignment {
  ReassignmentAfterUnsafetyAssignment() {
    int a = 10;
    m_i = a;
    m_i = 1;
  }
  int m_i;
};

namespace PR70189 {
#define RGB(r,g,b) ((unsigned long)(((unsigned char)(r)|((unsigned short)((unsigned char)(g))<<8))|(((unsigned long)(unsigned char)(b))<<16)))
#define INVALID_HANDLE_VALUE ((void*)(unsigned long long)-1)
#define SIMPLE 12

class Foo {
public:
  Foo() {
// CHECK-FIXES: Foo() : m_color(RGB(255, 128, 0)), m_handle(INVALID_HANDLE_VALUE), m_myval(SIMPLE) {
    m_color = RGB(255, 128, 0);
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'm_color' should be initialized in a member initializer of the constructor [cppcoreguidelines-prefer-member-initializer]
// CHECK-FIXES: {{^\ *$}}
    m_handle = INVALID_HANDLE_VALUE;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'm_handle' should be initialized in a member initializer of the constructor [cppcoreguidelines-prefer-member-initializer]
// CHECK-FIXES: {{^\ *$}}
    m_myval = SIMPLE;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'm_myval' should be initialized in a member initializer of the constructor [cppcoreguidelines-prefer-member-initializer]
// CHECK-FIXES: {{^\ *$}}
  }
private:
  unsigned long m_color;
  void* m_handle;
  int m_myval;
};

#undef SIMPLE
#undef INVALID_HANDLE_VALUE
#undef RGB
}

namespace GH77684 {
struct S1 {
// CHECK-MESSAGES: :[[@LINE+1]]:16: warning: 'M' should be initialized in a member initializer of the constructor [cppcoreguidelines-prefer-member-initializer]
  S1() : M{} { M = 0; }
// CHECK-FIXES:  S1() : M{0} { }
  int M;
};
struct S2 {
// CHECK-MESSAGES: :[[@LINE+1]]:17: warning: 'M' should be initialized in a member initializer of the constructor [cppcoreguidelines-prefer-member-initializer]
  S2() : M{2} { M = 1; }
// CHECK-FIXES:  S2() : M{1} { }
  int M;
};
struct T { int a; int b; int c; };
T v;
struct S3 {
// CHECK-MESSAGES: :[[@LINE+1]]:21: warning: 'M' should be initialized in a member initializer of the constructor [cppcoreguidelines-prefer-member-initializer]
  S3() : M{1,2,3} { M = v; }
// CHECK-FIXES:  S3() : M{v} { }
  T M;
};
}

namespace GH82970 {
struct InitFromBindingDecl {
  int m;
  InitFromBindingDecl() {
    struct { int i; } a;
    auto [n] = a;
    m = n;
  }
};
} // namespace GH82970

struct A {
  int m;
};

struct B : A {
  B() { m = 0; }
};

template <class T>
struct C : A {
  C() { m = 0; }
};
