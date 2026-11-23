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

  if (b) hoo: {
    return;
  } else loo: [[unlikely]]
    return;
  // CHECK-MESSAGES: :[[@LINE-2]]:9: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: } else { loo: {{[[][[]}}unlikely{{[]][]]}}
  // CHECK-FIXES-NEXT:   return;
  // CHECK-FIXES-NEXT: }

  if (b)
    return;
  else coo: [[unlikely]] {
    return;
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:9: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: if (b) {
  // CHECK-FIXES-NEXT:   return;
  // CHECK-FIXES-NEXT: } else coo: {{[[][[]}}unlikely{{[]][]]}} {

  if (b) aoo:
    return;
  else boo: [[unlikely]] {
    return;
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:9: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: if (b) { aoo:
  // CHECK-FIXES-NEXT:   return;
  // CHECK-FIXES-NEXT: } else boo: {{[[][[]}}unlikely{{[]][]]}} {

  if (b) moo: [[unlikely]]
    return;
  else noo: [[unlikely]] {
    return;
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:9: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: if (b) { moo: {{[[][[]}}unlikely{{[]][]]}}
  // CHECK-FIXES-NEXT:   return;
  // CHECK-FIXES-NEXT: } else noo: {{[[][[]}}unlikely{{[]][]]}} {

  if (b) poo: [[likely]] {
    return;
  } else qoo: [[unlikely]]
    return;
  // CHECK-MESSAGES: :[[@LINE-2]]:9: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: } else { qoo: {{[[][[]}}unlikely{{[]][]]}}
  // CHECK-FIXES-NEXT:    return;
  // CHECK-FIXES-NEXT: }
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

  if (b) {
    return;
  } else goo: [[unlikely]] {
    return;
  }

  if (b) roo: [[unlikely]] {
    return;
  } else {
    return;
  }

  if (b) poo: [[likely]]
    return;
  else qoo: [[unlikely]]
    return;
}
