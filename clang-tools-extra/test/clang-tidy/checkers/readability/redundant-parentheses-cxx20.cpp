// RUN: %check_clang_tidy -std=c++20-or-later %s readability-redundant-parentheses %t

struct C {
  int n;
};

// A class-type template parameter object is an lvalue, so the parentheses
// select 'const C &' over 'C' in both the template and its instantiation.
template <C N> decltype(auto) returnParameter() { return ((N)); }
// CHECK-MESSAGES: :[[@LINE-1]]:59: warning: redundant parentheses around expression [readability-redundant-parentheses]
// CHECK-FIXES:    template <C N> decltype(auto) returnParameter() { return (N); }
template <C N> decltype(auto) initParameter() {
  decltype(auto) v = ((N));
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    decltype(auto) v = (N);
  return v;
}
template <C N> decltype(auto) decltypeParameter() {
  decltype(((N))) v = N;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    decltype((N)) v = N;
  return v;
}

static_assert(__is_same(decltype(returnParameter<C{1}>()), const C &));
static_assert(__is_same(decltype(initParameter<C{1}>()), const C &));
static_assert(__is_same(decltype(decltypeParameter<C{1}>()), const C &));

// Substituting a reference template parameter into a template argument wraps
// it in synthetic parentheses, e.g. 'Array<char, (PATTERN).size() * 10>'.
struct StringView {
  constexpr unsigned long size() const { return 8; }
};
template <class T, unsigned long N> struct Array {
  T Data[N];
};
template <const StringView &Pattern, unsigned long Repeat>
struct RepeatPattern {
  static auto impl() { return Array<char, Pattern.size() * Repeat>{}; }
};
constexpr StringView PATTERN{};
auto RepeatedArray = RepeatPattern<PATTERN, 10>::impl();
