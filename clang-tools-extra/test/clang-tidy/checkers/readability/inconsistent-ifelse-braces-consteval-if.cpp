// RUN: %check_clang_tidy -std=c++23-or-later %s readability-inconsistent-ifelse-braces %t

bool cond(const char *) { return false; }
void do_something(const char *) {}

// Positive tests.
void f() {
  if consteval {
    if (cond("if1"))
      do_something("if-single-line");
    else {
    }
    // CHECK-MESSAGES: :[[@LINE-4]]:21: warning: statement should have braces [readability-inconsistent-ifelse-braces]
    // CHECK-FIXES: if (cond("if1")) {
    // CHECK-FIXES: } else {
  }

  if consteval {
    if (cond("if2"))
      do_something("if-single-line");
    else {
    }
    // CHECK-MESSAGES: :[[@LINE-4]]:21: warning: statement should have braces [readability-inconsistent-ifelse-braces]
    // CHECK-FIXES: if (cond("if2")) {
    // CHECK-FIXES: } else {
  } else {
    if (cond("if2.1")) {
    } else
      do_something("else-single-line");
    // CHECK-MESSAGES: :[[@LINE-2]]:11: warning: statement should have braces [readability-inconsistent-ifelse-braces]
    // CHECK-FIXES: } else {
    // CHECK-FIXES: }
  }
}

// Negative tests.
void g() {
  if consteval {
    if (cond("if0")) {
      do_something("if-single-line");
    } else if (cond("if0")) {
      do_something("elseif-single-line");
    } else {
      do_something("else-single-line");
    }
  } else {
    if (cond("if0.1"))
      do_something("if-single-line");
    else if (cond("if0.1"))
      do_something("elseif-single-line");
    else
      do_something("else-single-line");
  }
}
