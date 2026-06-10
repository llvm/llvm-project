// RUN: %check_clang_tidy -std=c++98-or-later %s readability-inconsistent-ifelse-braces %t

// Positive tests.
void f(bool b) {
  if (b) goo:
    return;
  else too: {
    return;
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:9: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: if (b) { goo:
  // CHECK-FIXES-NEXT:     return;
  // CHECK-FIXES-NEXT: } else too: {

  if (b) xoo: {
    return;
  } else yoo:
    return;
  // CHECK-MESSAGES: :[[@LINE-2]]:9: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: } else { yoo:
  // CHECK-FIXES-NEXT:   return;
  // CHECK-FIXES-NEXT: }

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
  } else foo: {
    return;
  }

  if (b) {
    return;
  } else goo: [[unlikely]] {
    return;
  }

  if (b) {
    return;
  } else hoo: {
    return;
  }

  if (b) joo: {
    return;
  } else {
    return;
  }

  if (b) roo: [[unlikely]] {
    return;
  } else {
    return;
  }

  if (b) koo: {
    return;
  } else loo: {
    return;
  }

  if (b) moo:
    return;
  else noo:
    return;

  if (b)
    return;
  else ooo:
    return;

  if (b) poo: [[likely]]
    return;
  else qoo: [[unlikely]]
    return;
}
