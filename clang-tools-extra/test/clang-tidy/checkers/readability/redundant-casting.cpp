// RUN: %check_clang_tidy -std=c++11-or-later %s readability-redundant-casting %t -- -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -std=c++11-or-later -check-suffix=,MACROS %s readability-redundant-casting %t -- \
// RUN:   -config='{CheckOptions: { readability-redundant-casting.IgnoreMacros: false }}' \
// RUN:   -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -std=c++11-or-later -check-suffix=,ALIASES %s readability-redundant-casting %t -- \
// RUN:   -config='{CheckOptions: { readability-redundant-casting.IgnoreTypeAliases: true }}' \
// RUN:   -- -fno-delayed-template-parsing

struct A {};
struct B : A {};
A getA();

void testRedundantStaticCasting(A& value) {
  A& a1 = static_cast<A&>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant explicit casting to the same type 'A' as the sub-expression, remove this casting [readability-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-3]]:36: note: source type originates from referencing this parameter
  // CHECK-FIXES: {{^}}  A& a1 = value;
}

void testRedundantConstCasting1(A& value) {
  A& a2 = const_cast<A&>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant explicit casting to the same type 'A' as the sub-expression, remove this casting [readability-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-3]]:36: note: source type originates from referencing this parameter
  // CHECK-FIXES: {{^}}  A& a2 = value;
}

void testRedundantConstCasting2(const A& value) {
  const A& a3 = const_cast<const A&>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: redundant explicit casting to the same type 'const A' as the sub-expression, remove this casting [readability-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-3]]:42: note: source type originates from referencing this parameter
  // CHECK-FIXES: {{^}}  const A& a3 = value;
}

void testRedundantReinterpretCasting(A& value) {
  A& a4 = reinterpret_cast<A&>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant explicit casting to the same type 'A' as the sub-expression, remove this casting [readability-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-3]]:41: note: source type originates from referencing this parameter
  // CHECK-FIXES: {{^}}  A& a4 = value;
}

void testRedundantCCasting(A& value) {
  A& a5 = (A&)(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant explicit casting to the same type 'A' as the sub-expression, remove this casting [readability-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-3]]:31: note: source type originates from referencing this parameter
  // CHECK-FIXES: {{^}}  A& a5 = value;
}

void testDoubleCasting(A& value) {
  A& a6 = static_cast<A&>(reinterpret_cast<A&>(value));
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant explicit casting to the same type 'A' as the sub-expression, remove this casting [readability-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:27: warning: redundant explicit casting to the same type 'A' as the sub-expression, remove this casting [readability-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-4]]:27: note: source type originates from referencing this parameter
  // CHECK-FIXES: {{^}}  A& a6 = value;
}

void testDiffrentTypesCast(B& value) {
  A& a7 = static_cast<A&>(value);
}

void testCastingWithAuto() {
  auto a = getA();
  A& a8 = static_cast<A&>(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant explicit casting to the same type 'A' as the sub-expression, remove this casting [readability-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-3]]:8: note: source type originates from referencing this variable
  // CHECK-FIXES: {{^}}  A& a8 = a;
}

void testCastingWithConstAuto() {
  const auto a = getA();
  const A& a9 = static_cast<const A&>(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: redundant explicit casting to the same type 'const A' as the sub-expression, remove this casting [readability-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-3]]:14: note: source type originates from referencing this variable
  // CHECK-FIXES: {{^}}  const A& a9 = a;
}

void testCastingWithAutoPtr(A& ptr) {
  auto* a = &ptr;
  A* a10 = static_cast<A*>(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant explicit casting to the same type 'A *' as the sub-expression, remove this casting [readability-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-3]]:9: note: source type originates from referencing this variable
  // CHECK-FIXES: {{^}}  A* a10 = a;
}

template<typename T>
void testRedundantTemplateCasting(T& value) {
  A& a = static_cast<A&>(value);
  T& t = static_cast<T&>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: redundant explicit casting to the same type 'T' as the sub-expression, remove this casting [readability-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-4]]:38: note: source type originates from referencing this parameter
  // CHECK-FIXES: {{^}}  T& t = value;
}

void testTemplate() {
  A value;
  testRedundantTemplateCasting(value);
}

void testValidRefConstCast() {
  const auto a = getA();
  A& a11 = const_cast<A&>(a);
}

void testValidPtrConstCast(const A* ptr) {
  A* a12 = const_cast<A*>(ptr);
}

#define CAST(X) static_cast<int>(X)

void testMacroCasting(int value) {
  int a = CAST(value);
  // CHECK-MESSAGES-MACROS: :[[@LINE-1]]:11: warning: redundant explicit casting to the same type 'int' as the sub-expression, remove this casting [readability-redundant-casting]
}

#define PTR_NAME name

void testMacroCasting(A* PTR_NAME) {
  A* a13 = static_cast<A*>(PTR_NAME);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant explicit casting to the same type 'A *' as the sub-expression, remove this casting [readability-redundant-casting]
  // CHECK-FIXES: {{^}}  A* a13 = PTR_NAME;
}

struct CastBool {
  operator bool() const {
    return true;
  }
};

void testUserOperatorCast(const CastBool& value) {
  bool b = static_cast<bool>(value);
}

using TypeA = A;

void testTypedefCast(A& value) {
  TypeA& a = static_cast<TypeA&>(value);
  // CHECK-MESSAGES-ALIASES: :[[@LINE-1]]:14: warning: redundant explicit casting to the same type 'TypeA' (aka 'A') as the sub-expression, remove this casting [readability-redundant-casting]
  // CHECK-FIXES-ALIASES: {{^}}  TypeA& a = value;
}

void testTypedefCast2(TypeA& value) {
  A& a = static_cast<A&>(value);
  // CHECK-MESSAGES-ALIASES: :[[@LINE-1]]:10: warning: redundant explicit casting to the same type 'A' as the sub-expression, remove this casting [readability-redundant-casting]
  // CHECK-FIXES-ALIASES: {{^}}  A& a = value;
}

void testFunctionalCastWithPrimitive(int a) {
  int b = int(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant explicit casting to the same type 'int' as the sub-expression, remove this casting [readability-redundant-casting]
  // CHECK-FIXES: {{^}}  int b = a;
}

void testFunctionalCastWithInitExpr(unsigned a) {
  unsigned b = ~unsigned{!a};
  unsigned c = unsigned{0};
}

void testBinaryOperator(char c) {
  int a = int(c - 'C');
}

struct BIT {
  bool b:1;
};

template<typename ...Args>
void make(Args&& ...);

void testBinaryOperator(BIT b) {
  make((bool)b.b);
}

struct Class {
  using Iterator = const char*;

  Iterator begin() {
    return static_cast<Iterator>(first());
// CHECK-MESSAGES-ALIASES: :[[@LINE-1]]:12: warning: redundant explicit casting to the same type 'Iterator' (aka 'const char *') as the sub-expression, remove this casting [readability-redundant-casting]
// CHECK-MESSAGES-ALIASES: :[[@LINE+4]]:15: note: source type originates from the invocation of this method
// CHECK-FIXES-ALIASES: {{^}}    return first();
  }

  const char* first();
};

void testAddOperation(int aa, int bb) {
  int c = static_cast<int>(aa + bb) * aa;
 // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant explicit casting to the same type 'int' as the sub-expression, remove this casting [readability-redundant-casting]
 // CHECK-FIXES: {{^}}  int c = (aa + bb) * aa;
}

void testAddOperationWithParen(int a, int b) {
  int c = static_cast<int>((a+b))*a;
 // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant explicit casting to the same type 'int' as the sub-expression, remove this casting [readability-redundant-casting]
 // CHECK-FIXES: {{^}}  int c = (a+b)*a;
}

void testRValueCast(int&& a) {
  int&& b = static_cast<int&&>(a);
  int&& c = static_cast<int&&>(10);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: redundant explicit casting to the same type 'int' as the sub-expression, remove this casting [readability-redundant-casting]
  // CHECK-FIXES: {{^}}  int&& c = 10;
}

template <int V>
void testRedundantNTTPCasting() {
  int a = static_cast<int>(V);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant explicit casting to the same type 'int' as the sub-expression, remove this casting [readability-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-4]]:15: note: source type originates from referencing this non-type template parameter
  // CHECK-FIXES: {{^}}  int a = V;
}

template <typename T, T V>
void testValidNTTPCasting() {
  int a = static_cast<int>(V);
}

template <typename T, T V>
void testRedundantDependentNTTPCasting() {
  T a = static_cast<T>(V);
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: redundant explicit casting to the same type 'T' as the sub-expression, remove this casting [readability-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-4]]:25: note: source type originates from referencing this non-type template parameter
  // CHECK-FIXES: {{^}}  T a = V;
}
