// RUN: %check_clang_tidy -std=c++17 %s modernize-use-as-const %t

// CHECK-FIXES: #include <utility>

struct S {};
struct Derived : S {};
void use(const S &);

typedef S SAlias;

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

// The operand type may be spelled through a typedef.
void via_typedef(SAlias obj) {
  use(static_cast<const S &>(obj));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use 'std::as_const' instead of 'static_cast' to add 'const' [modernize-use-as-const]
  // CHECK-FIXES: use(std::as_const(obj));
}

// A concrete cast inside a template still fires, and an instantiation must not
// report it a second time.
template <typename T>
struct Holder {
  const S &get(S &s) {
    return static_cast<const S &>(s);
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use 'std::as_const' instead of 'static_cast' to add 'const' [modernize-use-as-const]
    // CHECK-FIXES: return std::as_const(s);
  }
};
template struct Holder<int>;

void negatives(const S cobj, Derived d, S obj) {
  // Already const: there is nothing to add.
  use(static_cast<const S &>(cobj));
  // Derived-to-base changes the type, not just adds const.
  use(static_cast<const S &>(d));
  // Not a cast to a const reference.
  S copy = static_cast<S>(obj);
  (void)copy;
}

// A cast written inside a macro is left untouched: the fix would be unreliable.
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
