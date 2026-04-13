// RUN: %check_clang_tidy %s readability-qualified-auto %t --
// RUN: %check_clang_tidy %s readability-qualified-auto %t -check-suffix=NOCONST \
// RUN: -config='{CheckOptions: { readability-qualified-auto.AddConstToQualified: false }}' --

const int *getCIntPtr();

void foo() {
  auto *QualCPtr = getCIntPtr();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto *QualCPtr' can be declared as 'const auto *QualCPtr'
  // CHECK-FIXES: const auto *QualCPtr = getCIntPtr();
  // No warning for NOCONST
}
