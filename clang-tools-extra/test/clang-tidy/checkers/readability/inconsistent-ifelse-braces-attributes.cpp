// RUN: %check_clang_tidy  -std=c++20-or-later %s readability-inconsistent-ifelse-braces %t

// Positive tests.
void f(bool b) {
  if (b) [[likely]] return;
  else {
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:20: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: if (b) {{[[][[]}}likely{{[]][]]}} { return;
  // CHECK-FIXES: } else {

  if (b) {
  } else [[unlikely]]
    return;
  // CHECK-MESSAGES: :[[@LINE-2]]:22: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: } else {{[[][[]}}unlikely{{[]][]]}} {

  if (b) [[likely]] {
  } else [[unlikely]]
    return;
  // CHECK-MESSAGES: :[[@LINE-2]]:22: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: } else {{[[][[]}}unlikely{{[]][]]}} {

  if (b) [[likely]]
    return;
  else [[unlikely]] {
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:20: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: if (b) {{[[][[]}}likely{{[]][]]}} {
  // CHECK-FIXES: } else {{[[][[]}}unlikely{{[]][]]}} {
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

  if (b) [[likely]] {
    return;
  } else {
    return;
  }

  if (b) {
    return;
  } else [[unlikely]] {
    return;
  }

  if (b) [[likely]] {
    return;
  } else [[unlikely]] {
    return;
  }

  if (b) [[likely]] {
    return;
  } else if (b) [[unlikely]] {
    return;
  } else {
    return;
  }

  if (b) [[likely]] [[likely]] {
    return;
  } else [[unlikely]] [[unlikely]] {
    return;
  }
}
