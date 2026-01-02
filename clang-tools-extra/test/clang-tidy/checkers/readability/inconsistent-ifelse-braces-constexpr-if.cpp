// RUN: %check_clang_tidy -std=c++17-or-later %s readability-inconsistent-ifelse-braces %t

constexpr bool cond(const char *) { return false; }
constexpr void do_something(const char *) {}

// Positive tests.
void f() {
  if constexpr (cond("if0") /*comment*/) do_something("if-same-line");
  else {
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:41: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: if constexpr (cond("if0") /*comment*/) { do_something("if-same-line");
  // CHECK-FIXES: } else {

  if constexpr (cond("if0.1") /*comment*/) {
  } else do_something("else-same-line");
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: } else { do_something("else-same-line");
  // CHECK-FIXES: }

  if constexpr (cond("if1"))
    do_something("if-single-line");
  else {
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:29: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: if constexpr (cond("if1")) {
  // CHECK-FIXES: } else {

  if constexpr (cond("if1.1")) {
  } else
    do_something("else-single-line");
  // CHECK-MESSAGES: :[[@LINE-2]]:9: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: } else {
  // CHECK-FIXES: }

  if constexpr (cond("if2") /*comment*/)
    // some comment
    do_something("if-multi-line");
  else {
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:41: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: if constexpr (cond("if2") /*comment*/) {
  // CHECK-FIXES: } else {

  if constexpr (cond("if2.1") /*comment*/) {
  } else
    // some comment
    do_something("else-multi-line");
  // CHECK-MESSAGES: :[[@LINE-3]]:9: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: } else {
  // CHECK-FIXES: }

  if constexpr (cond("if3")) do_something("elseif-same-line");
  else if constexpr (cond("if3")) {
  } else {
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:29: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: if constexpr (cond("if3")) { do_something("elseif-same-line");
  // CHECK-FIXES: } else if constexpr (cond("if3")) {

  if constexpr (cond("if3.1")) {
  } else if constexpr (cond("if3.1")) do_something("elseif-same-line");
  else {
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:38: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: } else if constexpr (cond("if3.1")) { do_something("elseif-same-line");
  // CHECK-FIXES: } else {

  if constexpr (cond("if3.2")) {
  } else if constexpr (cond("if3.2")) {
  } else do_something("else-same-line");
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: } else { do_something("else-same-line");
  // CHECK-FIXES: }

  if constexpr (cond("if4-outer"))
    if constexpr (cond("if4-inner"))
      do_something("if-single-line");
    else {
    }
  else {
  }
  // CHECK-MESSAGES: :[[@LINE-7]]:35: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-MESSAGES: :[[@LINE-7]]:37: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: if constexpr (cond("if4-outer")) {
  // CHECK-FIXES: if constexpr (cond("if4-inner")) {
  // CHECK-FIXES: } else {
  // CHECK-FIXES: } else {

  if constexpr (cond("if5"))
      do_something("if-single-line");
  else if constexpr (cond("if5")) {
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:29: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: if constexpr (cond("if5")) {
  // CHECK-FIXES: } else if constexpr (cond("if5")) {

  if constexpr (cond("if5.1")) {
  } else if constexpr (cond("if5.1"))
      do_something("elseif-single-line");
  // CHECK-MESSAGES: :[[@LINE-2]]:38: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: } else if constexpr (cond("if5.1")) {
  // CHECK-FIXES: }
}

// Negative tests.
void g() {
  if constexpr (cond("if0")) {
    do_something("if-single-line");
  } else if constexpr (cond("if0")) {
    do_something("elseif-single-line");
  } else {
    do_something("else-single-line");
  }

  if constexpr (cond("if1"))
    do_something("if-single-line");
  else if constexpr (cond("if1"))
    do_something("elseif-single-line");
  else
    do_something("else-single-line");
}
