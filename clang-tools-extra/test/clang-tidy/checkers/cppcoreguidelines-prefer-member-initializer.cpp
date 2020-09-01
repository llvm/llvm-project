// RUN: %check_clang_tidy %s cppcoreguidelines-prefer-member-initializer %t -- -- -fcxx-exceptions

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
    // CHECK-FIXES: Complex19(int) : n(12) {
    n = 12;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'n' should be initialized in a member initializer of the constructor [cppcoreguidelines-prefer-member-initializer]
    // CHECK-FIXES: {{^\ *$}}
  }

  ~Complex19() = default;
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
