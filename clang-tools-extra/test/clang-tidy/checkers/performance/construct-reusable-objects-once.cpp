// RUN: %check_clang_tidy %s performance-construct-reusable-objects-once %t -- \
// RUN:   -config='{CheckOptions: { \
// RUN:     performance-construct-reusable-objects-once.CheckedClasses: "::std::basic_regex;ReusableClass", \
// RUN:     performance-construct-reusable-objects-once.IgnoredFunctions: "::main;::global::init;C::C;D::~D;ns::MyClass::foo", \
// RUN:   }}' \
// RUN: -- -fno-delayed-template-parsing -I%S/Inputs/construct-reusable-objects-once

#include "regex.h"

void PositiveEmpty() {
  const std::regex r1;
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: variable 'r1' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'PositiveEmpty';
  const std::wregex wr1;
  // CHECK-MESSAGES: [[@LINE-1]]:21: warning: variable 'wr1' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'PositiveEmpty';
}

void PositiveSugared() {
  using my_using_regex = std::regex;
  const my_using_regex r1;
  // CHECK-MESSAGES: [[@LINE-1]]:24: warning: variable 'r1' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'PositiveSugared';

  typedef std::wregex my_typedef_regex;
  const my_typedef_regex wr1;
  // CHECK-MESSAGES: [[@LINE-1]]:26: warning: variable 'wr1' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'PositiveSugared';

  using std::regex;
  const regex r2;
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: variable 'r2' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'PositiveSugared';
}

void PositiveWithLiterals() {
  const std::regex r1("x");
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: variable 'r1' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'PositiveWithLiterals';
  const std::wregex wr1(L"x");
  // CHECK-MESSAGES: [[@LINE-1]]:21: warning: variable 'wr1' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'PositiveWithLiterals';
  const std::regex r2("x", 1);
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: variable 'r2' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'PositiveWithLiterals';
  const std::wregex wr2(L"x", 2);
  // CHECK-MESSAGES: [[@LINE-1]]:21: warning: variable 'wr2' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'PositiveWithLiterals';
}

const char* const text = "";
const char text2[] = "";

const wchar_t* const wtext = L"";
const wchar_t wtext2[] = L"";

const int c_i = 1;
constexpr int ce_i = 1;

void PositiveWithVariables() {
  const std::regex r1("x", c_i);
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: variable 'r1' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'PositiveWithVariables';
  const std::regex r2("x", ce_i);
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: variable 'r2' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'PositiveWithVariables';

  const std::regex r3(text);
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: variable 'r3' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'PositiveWithVariables';
  const std::wregex wr3(wtext);
  // CHECK-MESSAGES: [[@LINE-1]]:21: warning: variable 'wr3' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'PositiveWithVariables';
  const std::regex r4(text2);
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: variable 'r4' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'PositiveWithVariables';
  const std::wregex wr4(wtext2);
  // CHECK-MESSAGES: [[@LINE-1]]:21: warning: variable 'wr4' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'PositiveWithVariables';

  const std::regex r5(text, 1);
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: variable 'r5' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'PositiveWithVariables';
  const std::wregex wr5(wtext, 2);
  // CHECK-MESSAGES: [[@LINE-1]]:21: warning: variable 'wr5' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'PositiveWithVariables';
  const std::regex r6(text2, 3);
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: variable 'r6' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'PositiveWithVariables';
  const std::wregex wr6(wtext2, 4);
  // CHECK-MESSAGES: [[@LINE-1]]:21: warning: variable 'wr6' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'PositiveWithVariables';

  // Local variables constructed from literals
  const char* const ltext = "";
  const wchar_t* const wltext = L"";
  const int l_i = 1;
  constexpr int l_ce_i = 1;

  const std::regex r7(ltext, 7);
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: variable 'r7' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'PositiveWithVariables';
  const std::wregex wr7(wltext, 8);
  // CHECK-MESSAGES: [[@LINE-1]]:21: warning: variable 'wr7' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'PositiveWithVariables';

  const std::regex r8(text, 1, std::regex::icase);
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: variable 'r8' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'PositiveWithVariables';
  const std::wregex wr8(wtext, 2, std::regex::icase);
  // CHECK-MESSAGES: [[@LINE-1]]:21: warning: variable 'wr8' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'PositiveWithVariables';

  const std::regex r9(text, l_i, std::regex::icase);
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: variable 'r9' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'PositiveWithVariables';
  const std::regex r10(text, l_ce_i, std::regex::icase);
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: variable 'r10' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'PositiveWithVariables';
}

void PositiveNested() {
  if (true) {
    const int i = 0;
    while (true) {
      const std::regex r1("x", i);
      // CHECK-MESSAGES: [[@LINE-1]]:24: warning: variable 'r1' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'PositiveNested';
    }
  }
}

void PositiveFromTemporary() {
  const auto r1 = std::regex("x");
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: variable 'r1' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'PositiveFromTemporary';
  const auto wr1 = std::wregex(L"x", 0, std::regex::ECMAScript);
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: variable 'wr1' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'PositiveFromTemporary';
}

static void PositiveStaticFunction() {
  const int l_i = 1;
  constexpr int l_ce_i = 1;

  const std::regex r1(text, l_i, std::regex::icase);
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: variable 'r1' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'PositiveStaticFunction';
  const std::regex r2(text, l_ce_i, std::regex::icase);
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: variable 'r2' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'PositiveStaticFunction';
}

class PositiveClass {
  PositiveClass() {
    const std::regex r1("text");
    // CHECK-MESSAGES: [[@LINE-1]]:22: warning: variable 'r1' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'PositiveClass::PositiveClass';
  }

  void foo() {
    const std::regex r2("text");
    // CHECK-MESSAGES: [[@LINE-1]]:22: warning: variable 'r2' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'PositiveClass::foo';
  }

  static void bar() {
    const std::regex r3("text");
    // CHECK-MESSAGES: [[@LINE-1]]:22: warning: variable 'r3' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'PositiveClass::bar';
  }

};

// All enum variations

enum En1 {
  e1
};

enum class En2 {
  e2
};

enum class En3 : short {
  e3
};

enum En4 : short {
  e4
};

template <typename T>
struct ReusableClass {
  ReusableClass(float f) {}
  ReusableClass(const char* ptr) {}
  ReusableClass(int i) {}
  ReusableClass(En1 e1) {}
  ReusableClass(En2 e2) {}
  ReusableClass(En3 e3) {}
  ReusableClass(En4 e4) {}
};

void PositiveAllLiteralTypes() {
  const ReusableClass<int> rc1(1);
  // CHECK-MESSAGES: [[@LINE-1]]:28: warning: variable 'rc1' of type 'ReusableClass' is constructed with only constant literals on each invocation of 'PositiveAllLiteralTypes';
  const ReusableClass<int> rc2(1.0f);
  // CHECK-MESSAGES: [[@LINE-1]]:28: warning: variable 'rc2' of type 'ReusableClass' is constructed with only constant literals on each invocation of 'PositiveAllLiteralTypes';
  const ReusableClass<int> rc3("x");
  // CHECK-MESSAGES: [[@LINE-1]]:28: warning: variable 'rc3' of type 'ReusableClass' is constructed with only constant literals on each invocation of 'PositiveAllLiteralTypes';
  const ReusableClass<int> rc4(e1);
  // CHECK-MESSAGES: [[@LINE-1]]:28: warning: variable 'rc4' of type 'ReusableClass' is constructed with only constant literals on each invocation of 'PositiveAllLiteralTypes';
  const ReusableClass<int> rc5(En2::e2);
  // CHECK-MESSAGES: [[@LINE-1]]:28: warning: variable 'rc5' of type 'ReusableClass' is constructed with only constant literals on each invocation of 'PositiveAllLiteralTypes';
  const ReusableClass<int> rc6(En3::e3);
  // CHECK-MESSAGES: [[@LINE-1]]:28: warning: variable 'rc6' of type 'ReusableClass' is constructed with only constant literals on each invocation of 'PositiveAllLiteralTypes';
  const ReusableClass<int> rc7(e4);
  // CHECK-MESSAGES: [[@LINE-1]]:28: warning: variable 'rc7' of type 'ReusableClass' is constructed with only constant literals on each invocation of 'PositiveAllLiteralTypes';

  using my_enum = En4;
  const ReusableClass<int> rc8(my_enum::e4);
  // CHECK-MESSAGES: [[@LINE-1]]:28: warning: variable 'rc8' of type 'ReusableClass' is constructed with only constant literals on each invocation of 'PositiveAllLiteralTypes';

  const En2 my_e2 = En2::e2;
  const ReusableClass<int> rc9(my_e2);
  // CHECK-MESSAGES: [[@LINE-1]]:28: warning: variable 'rc9' of type 'ReusableClass' is constructed with only constant literals on each invocation of 'PositiveAllLiteralTypes';
}

template <typename T>
void PositiveTemplated() {
  const ReusableClass<int> rc1(1); 
  // CHECK-MESSAGES: [[@LINE-1]]:28: warning: variable 'rc1' of type 'ReusableClass' is constructed with only constant literals on each invocation of 'PositiveTemplated'
}

// Negative cases

void NegativeNonConst() {
  std::regex r1("x");
  ReusableClass<int> rc1(1);
}

int mi = 0;
auto fl = std::regex::basic;

// can not detect this case since 'std::regex::basic' is already a const variable,
// and we only match variables constructed from literals
const auto fl2 = std::regex::basic;

void NegativeVariables() {
  int l_mi = 1;
  const auto l_fl = std::regex::ECMAScript;
  auto l_fl2 = std::regex::basic;
  
  const std::regex r1("x", mi);
  const std::regex r2("x", l_mi);
  const std::regex r3("x", 0, fl);
  const std::regex r4("x", 0, fl2);
  const std::regex r5("x", 0, l_fl);
  const std::regex r6("x", 0, l_fl2);
}

void NegativeOperators() {
  int a = 0;
  const std::regex r("x", a);

  const std::regex r2(r);
}

template <typename T>
void NegativeTemplated() {
  // currently we do not support ParenListExpr that is generated instead of ctor-call
  const ReusableClass<T> rc1(1);
}

struct RegexParamProvider {
  int get_int() { return 0; }
  const char* get_char() { return 0; }
  const char* operator[] (int) {
    return 0;
  }
};

void NegativeWithVariables() {
  int j = 1;
  const std::regex r1("x", j);

  auto l = std::regex::basic;
  const std::regex r2("x", 1, l);

  char* c;
  const std::regex r3(c, 1);

  RegexParamProvider p;

  const std::regex r4("text", p.get_int());
  const std::regex r5(p[1]);
  const std::regex r6(p.get_char());
}

void NegativeFromFunction1(int i) {
  const std::regex r("x", i);
}

void NegativeFromFunction2(const int i) {
  const std::regex r("x", i);
}

void NegativeFromFunction3(const char* c) {
  const std::regex r(c);
}

void NegativeFromFunction4(const char* c = "default") {
  const std::regex r(c);
}

// Negative global variables

const std::regex gr1("x");
const std::wregex wgr1(L"x");

static const std::regex sgr1("x");
static const std::wregex swgr1(L"x");

void NegativeStatic() {
  static const std::regex r("x");
  static const std::wregex wr(L"x");
  static const std::regex r1("x", 1);
  static const std::wregex wr1(L"x", 2);
}

class NegativeClass {
  NegativeClass() : r("x") {
    r = std::regex("x2");
    static const std::regex r2("x");
  }

  NegativeClass(const char* c) {
    const std::regex r2(c);
  }

  void NegativeMethod(const char* c = "some value") {
    const std::regex r3(c);
  }

  std::regex r;
};


// Ignored functions check

int main() {
  const std::regex r1("x");
}

namespace global {

void init() {
  const std::regex r("x");
}

} // namespace global 

void init() {
  const std::regex r("x");
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: variable 'r' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'init';
}

class C {
 public:
  C() {
    const std::regex r("x");
  }

  ~C() {
    const std::regex r("x");
    // CHECK-MESSAGES: [[@LINE-1]]:22: warning: variable 'r' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'C::~C';
  }
};

class D {
 public:
  D() {
    const std::regex r("x");
    // CHECK-MESSAGES: [[@LINE-1]]:22: warning: variable 'r' of type 'std::basic_regex' is constructed with only constant literals on each invocation of 'D::D';
  }

  ~D() {
    const std::regex r("x");
  }
};

namespace ns {

class MyClass {
 public:
  void foo() {
    const std::regex r("x");
  }
};

} // namespace ns
