// RUN: %check_clang_tidy %s readability-uppercase-literal-suffix %t -- \
// RUN:   -config="{CheckOptions: [{key: readability-uppercase-literal-suffix.IgnoreMacros, value: 0}]}" \
// RUN:   -- -I %S

void macros() {
#define INMACRO(X) 1.f
  static constexpr auto m1 = INMACRO();
  // CHECK-NOTES: :[[@LINE-1]]:30: warning: floating point literal has suffix 'f', which is not uppercase
  // CHECK-NOTES: :[[@LINE-3]]:20: note: expanded from macro 'INMACRO'
  // CHECK-FIXES: #define INMACRO(X) 1.f
  // CHECK-FIXES: static constexpr auto m1 = INMACRO();
  // ^ so no fix-its here.
}

void horrible_macros() {
#define MAKE_UNSIGNED(x) x##u
#define ONE MAKE_UNSIGNED(1)
  static constexpr auto hm0 = ONE;
  // CHECK-NOTES: :[[@LINE-1]]:31: warning: integer literal has suffix 'u', which is not uppercase
  // CHECK-NOTES: :[[@LINE-3]]:13: note: expanded from macro 'ONE'
  // CHECK-NOTES: :[[@LINE-5]]:26: note: expanded from macro 'MAKE_UNSIGNED'
  // CHECK-NOTES: note: expanded from here
  // CHECK-FIXES: #define MAKE_UNSIGNED(x) x##u
  // CHECK-FIXES: #define ONE MAKE_UNSIGNED(1)
  // CHECK-FIXES: static constexpr auto hm0 = ONE;
  // Certainly no fix-its.
}
