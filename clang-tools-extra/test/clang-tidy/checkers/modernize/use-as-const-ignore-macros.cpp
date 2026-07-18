// RUN: %check_clang_tidy -std=c++17-or-later %s modernize-use-as-const %t -- \
// RUN:   -config="{CheckOptions: {modernize-use-as-const.IgnoreMacros: false}}"

// CHECK-FIXES: #include <utility>

struct S {};
void use(const S &);

#define TO_CONST(x) static_cast<const S &>(x)

// Reported at the expansion, but not fixed.
void in_macro(S obj) {
  use(TO_CONST(obj));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use 'std::as_const' instead of 'static_cast' to add 'const' [modernize-use-as-const]
  // CHECK-FIXES: use(TO_CONST(obj));
}

void outside_macro(S obj) {
  use(static_cast<const S &>(obj));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use 'std::as_const' instead of 'static_cast' to add 'const' [modernize-use-as-const]
  // CHECK-FIXES: use(std::as_const(obj));
}
