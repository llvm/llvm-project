// RUN: %check_clang_tidy  -std=c++20-or-later %s readability-inconsistent-ifelse-braces %t

// Positive tests.
void f(bool b) {
  if (b) [[likely]] return;
  else {
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:9: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: if (b) { {{[[][[]}}likely{{[]][]]}} return;
  // CHECK-FIXES: } else {

  if (b) {
  } else [[unlikely]]
    return;
  // CHECK-MESSAGES: :[[@LINE-2]]:9: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: } else { {{[[][[]}}unlikely{{[]][]]}}
}

// Negative tests.
void g(bool b) {
  if (b) {
    return;
  }

  if (b) { [[likely]]
    return;
  }

  if (b) { [[unlikely]]
    return;
  } else { [[likely]]
    return;
  }

  if (b) [[likely]]
    return;

  if (b) [[unlikely]]
    return;
  else [[likely]]
    return;
}
