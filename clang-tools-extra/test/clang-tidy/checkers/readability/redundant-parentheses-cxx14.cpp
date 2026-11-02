// RUN: %check_clang_tidy -std=c++14-or-later %s readability-redundant-parentheses %t

int g = 0;

decltype(auto) returnDecltypeAuto() { return (g); }
auto returnTrailingDecltypeAuto() -> decltype(auto) { return (g); }
template <class T> decltype(auto) returnDecltypeAutoTemplate(T &t) { return (t); }
int &instantiate() { return returnDecltypeAutoTemplate(g); }

struct S {
  decltype(auto) member() { return (g); }
};

auto returnAuto() { return (g); }
// CHECK-MESSAGES: :[[@LINE-1]]:28: warning: redundant parentheses around expression [readability-redundant-parentheses]
// CHECK-FIXES:    auto returnAuto() { return g; }
int &returnReference() { return (g); }
// CHECK-MESSAGES: :[[@LINE-1]]:33: warning: redundant parentheses around expression [readability-redundant-parentheses]
// CHECK-FIXES:    int &returnReference() { return g; }
decltype(auto) returnLiteral() { return (1); }
// CHECK-MESSAGES: :[[@LINE-1]]:41: warning: redundant parentheses around expression [readability-redundant-parentheses]
// CHECK-FIXES:    decltype(auto) returnLiteral() { return 1; }

void decltypeAuto(int x) {
  decltype(auto) a = (x);
  decltype(auto) b((x));
  decltype(auto) (c) = (x);
  decltype(auto) d = ((x));
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    decltype(auto) d = (x);
  decltype(auto) e = (x) + 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    decltype(auto) e = x + 1;
  auto f = (x);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    auto f = x;
  auto &h = (x);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    auto &h = x;
  decltype(auto) i = (1);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    decltype(auto) i = 1;
  auto lambda = [&]() -> decltype(auto) { return (x); };
  auto autoLambda = [&]() { return (x); };
  // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    auto autoLambda = [&]() { return x; };
}

void listInit(int x) {
  decltype(auto) a{(x)};
  decltype(auto) b{((x))};
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    decltype(auto) b{(x)};
  decltype(auto) c{(1)};
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    decltype(auto) c{1};
}

struct D {
  int m;
  static int sm;
};
template <class T> int vt = 0;

// Dependent operands are matched both in the template and in its instantiation.
template <class T> decltype(auto) dependentMember(T &t) { return ((t.m)); }
// CHECK-MESSAGES: :[[@LINE-1]]:67: warning: redundant parentheses around expression [readability-redundant-parentheses]
// CHECK-FIXES:    template <class T> decltype(auto) dependentMember(T &t) { return (t.m); }
template <class T> decltype(auto) dependentStatic(T &) { return ((T::sm)); }
// CHECK-MESSAGES: :[[@LINE-1]]:66: warning: redundant parentheses around expression [readability-redundant-parentheses]
// CHECK-FIXES:    template <class T> decltype(auto) dependentStatic(T &) { return (T::sm); }
template <class T> decltype(auto) dependentTemplate() { return ((vt<T>)); }
// CHECK-MESSAGES: :[[@LINE-1]]:65: warning: redundant parentheses around expression [readability-redundant-parentheses]
// CHECK-FIXES:    template <class T> decltype(auto) dependentTemplate() { return (vt<T>); }
auto genericLambda = [](auto &t) -> decltype(auto) { return ((t.m)); };
// CHECK-MESSAGES: :[[@LINE-1]]:62: warning: redundant parentheses around expression [readability-redundant-parentheses]
// CHECK-FIXES:    auto genericLambda = [](auto &t) -> decltype(auto) { return (t.m); };

template <int N> decltype(auto) substitutedParameter() { return ((N)); }
// CHECK-MESSAGES: :[[@LINE-1]]:66: warning: redundant parentheses around expression [readability-redundant-parentheses]
// CHECK-FIXES:    template <int N> decltype(auto) substitutedParameter() { return (N); }

template <class T> void dependentInit(T &t) {
  decltype(((t.m))) a = t.m;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    decltype((t.m)) a = t.m;
  decltype(auto) b = ((t.m));
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    decltype(auto) b = (t.m);
  decltype(auto) c((t));
  decltype(auto) d{(t)};
}

void instantiate(D &d) {
  dependentMember(d);
  dependentStatic(d);
  dependentTemplate<int>();
  genericLambda(d);
  substitutedParameter<1>();
  dependentInit(d);
}

struct Temporary {
  Temporary();
  ~Temporary();
  int m;
  int value();
};
Temporary makeTemporary();

void temporaryInit() {
  decltype(auto) a = (Temporary().m);
  decltype(auto) b{(makeTemporary().m)};
  decltype(auto) c((makeTemporary().m));
  decltype(auto) d = ((makeTemporary().m));
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    decltype(auto) d = (makeTemporary().m);
  decltype(auto) e{((makeTemporary().m))};
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    decltype(auto) e{(makeTemporary().m)};
  decltype(auto) value = (makeTemporary().value());
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    decltype(auto) value = makeTemporary().value();
  static_assert(__is_same(decltype(a), int &&), "");
  static_assert(__is_same(decltype(b), int &&), "");
  static_assert(__is_same(decltype(c), int &&), "");
  static_assert(__is_same(decltype(d), int &&), "");
  static_assert(__is_same(decltype(e), int &&), "");
  static_assert(__is_same(decltype((Temporary().m)), int &&), "");
}

// Create a cleanup without returning a reference to a temporary (ill-formed in C++26).
Temporary &&getTemporary(Temporary);
decltype(auto) temporaryReturn() { return (getTemporary(Temporary()).m); }
decltype(auto) nestedTemporaryReturn() { return ((getTemporary(Temporary()).m)); }
// CHECK-MESSAGES: :[[@LINE-1]]:50: warning: redundant parentheses around expression [readability-redundant-parentheses]
// CHECK-FIXES:    decltype(auto) nestedTemporaryReturn() { return (getTemporary(Temporary()).m); }
template <class T> decltype(auto) dependentTemporaryReturn() { return ((getTemporary(T()).m)); }
// CHECK-MESSAGES: :[[@LINE-1]]:72: warning: redundant parentheses around expression [readability-redundant-parentheses]
// CHECK-FIXES:    template <class T> decltype(auto) dependentTemporaryReturn() { return (getTemporary(T()).m); }
static_assert(__is_same(decltype(temporaryReturn()), int &&), "");
static_assert(__is_same(decltype(nestedTemporaryReturn()), int &&), "");
static_assert(__is_same(decltype(dependentTemporaryReturn<Temporary>()), int &&), "");
