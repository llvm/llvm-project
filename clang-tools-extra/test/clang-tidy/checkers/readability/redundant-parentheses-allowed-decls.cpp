// RUN: %check_clang_tidy %s readability-redundant-parentheses %t \
// RUN: -config='{CheckOptions: {readability-redundant-parentheses.AllowedDecls: ""}}'

namespace std {
  template<class T> T max(T, T);
  template<class T> T min(T, T);
} // namespace std

void foo() {
  (std::max)(1, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES: std::max(1, 2);
  (std::min)(1, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES: std::min(1, 2);
}
