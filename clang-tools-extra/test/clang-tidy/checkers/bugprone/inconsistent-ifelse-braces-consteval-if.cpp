// RUN: %check_clang_tidy -std=c++23-or-later %s bugprone-inconsistent-ifelse-braces %t

bool cond(const char *) { return false; }
void do_something(const char *) {}

// Positive tests.
void f() {
  if consteval {
    if (cond("if1"))
      do_something("if-consteval-single-line");
    else {
    }
    // CHECK-MESSAGES: :[[@LINE-4]]:21: warning: <message> [bugprone-inconsistent-ifelse-braces]
    // CHECK-FIXES: if (cond("if1")) {
    // CHECK-FIXES: } else {
  } else {
    if (cond("if1.1")) {
    } else
      do_something("if-consteval-single-line");
    // CHECK-MESSAGES: :[[@LINE-2]]:11: warning: <message> [bugprone-inconsistent-ifelse-braces]
    // CHECK-FIXES: } else {
    // CHECK-FIXES: }
  }
}

// Negative tests.
void g() {
}
