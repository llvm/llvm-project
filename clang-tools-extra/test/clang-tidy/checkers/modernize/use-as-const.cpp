// RUN: %check_clang_tidy -std=c++17-or-later %s modernize-use-as-const %t

// CHECK-FIXES: #include <utility>

struct S {};
struct Derived : S {};
void use(const S &);
void use_cv(const volatile S &);

typedef S SAlias;
using ConstRef = const S &;

void basic(S obj) {
  use(static_cast<const S &>(obj));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use 'std::as_const' instead of 'static_cast' to add 'const' [modernize-use-as-const]
  // CHECK-FIXES: use(std::as_const(obj));
}

struct Wrap {
  S m;
};

void other_lvalues(Wrap w, S *p, S arr[1]) {
  use(static_cast<const S &>(w.m));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use 'std::as_const' instead of 'static_cast' to add 'const' [modernize-use-as-const]
  // CHECK-FIXES: use(std::as_const(w.m));
  use(static_cast<const S &>(*p));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use 'std::as_const' instead of 'static_cast' to add 'const' [modernize-use-as-const]
  // CHECK-FIXES: use(std::as_const(*p));
  use(static_cast<const S &>(arr[0]));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use 'std::as_const' instead of 'static_cast' to add 'const' [modernize-use-as-const]
  // CHECK-FIXES: use(std::as_const(arr[0]));
}

void via_typedef(SAlias obj) {
  use(static_cast<const S &>(obj));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use 'std::as_const' instead of 'static_cast' to add 'const' [modernize-use-as-const]
  // CHECK-FIXES: use(std::as_const(obj));
}

void destination_typedef(S obj) {
  use(static_cast<ConstRef>(obj));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use 'std::as_const' instead of 'static_cast' to add 'const' [modernize-use-as-const]
  // CHECK-FIXES: use(std::as_const(obj));
}

void keeps_volatile(volatile S vobj) {
  use_cv(static_cast<const volatile S &>(vobj));
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use 'std::as_const' instead of 'static_cast' to add 'const' [modernize-use-as-const]
  // CHECK-FIXES: use_cv(std::as_const(vobj));
}

// The instantiation must not report the cast a second time.
template <typename T>
struct Holder {
  const S &get(S &s) {
    return static_cast<const S &>(s);
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use 'std::as_const' instead of 'static_cast' to add 'const' [modernize-use-as-const]
    // CHECK-FIXES: return std::as_const(s);
  }
};
template struct Holder<int>;

S make();

// std::as_const is deleted for rvalues, even though binding one to 'const S &'
// is well-formed.
void rvalue_negatives(S obj) {
  use(static_cast<const S &>(S{}));
  use(static_cast<const S &>(make()));
  use(static_cast<const S &>(static_cast<S &&>(obj)));
}

void negatives(const S cobj, Derived d, S obj) {
  use(static_cast<const S &>(cobj));
  use(static_cast<const S &>(d));
  S copy = static_cast<S>(obj);
  (void)copy;
  S &&r = static_cast<S &&>(obj);
  (void)r;
}

// See use-as-const-ignore-macros.cpp for IgnoreMacros: false.
#define TO_CONST(x) static_cast<const S &>(x)
void in_macro(S obj) {
  use(TO_CONST(obj));
}

// Dependent casts are skipped: when T is a reference type 'const T &' collapses
// and adds no const, so std::as_const would not be equivalent.
template <typename T>
const T &as_const_tmpl(T &x) {
  return static_cast<const T &>(x);
}
void instantiate() {
  S s;
  as_const_tmpl(s);
  as_const_tmpl<S &>(s);
}
