// RUN: %check_clang_tidy %s readability-redundant-parentheses %t

void parenExpr() {
  1 + 1;
  (1 + 1);
  ((1 + 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    (1 + 1);
  (((1 + 1)));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-MESSAGES: :[[@LINE-2]]:4: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    (1 + 1);
  ((((1 + 1))));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-MESSAGES: :[[@LINE-2]]:4: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-MESSAGES: :[[@LINE-3]]:5: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    (1 + 1);
}

#define EXP (1 + 1)
#define PAREN(e) (e)
void parenExprWithMacro() {
  EXP; // 1
  (EXP); // 2
  ((EXP)); // 3
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    (EXP); // 3
  PAREN((1));
}

void constant() {
  (1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    1;
  (1.0);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    1.0;
  (true);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    true;
  (',');
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    ',';
  ("v4");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    "v4";
  (nullptr);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    nullptr;
}

void declRefExpr(int a) {
  (a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    a;
}

void exceptions() {
  sizeof(1);
  alignof(2);
  alignof((3));
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    alignof(3);
}

namespace std {
  template<class T> T max(T, T);
  template<class T> T min(T, T);
} // namespace std
void ignoreStdMaxMin() {
  (std::max)(1,2);
  (std::min)(1,2);
}
