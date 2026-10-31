// RUN: %check_clang_tidy -std=c++98-or-later %s readability-inconsistent-ifelse-braces %t

#define MACRO_COND(x) cond(x)
#define MACRO_FUN (void)0

bool cond(const char *) { return false; }
void do_something(const char *) {}

// Positive tests.
void f() {
  if (cond("if0") /*comment*/) do_something("if-same-line");
  else {
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:31: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: if (cond("if0") /*comment*/) { do_something("if-same-line");
  // CHECK-FIXES: } else {

  if (cond("if0.1") /*comment*/) {
  } else do_something("else-same-line");
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: } else { do_something("else-same-line");
  // CHECK-FIXES: }

  if (cond("if1"))
    do_something("if-single-line");
  else {
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:19: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: if (cond("if1")) {
  // CHECK-FIXES: } else {

  if (cond("if1.1")) {
  } else
    do_something("else-single-line");
  // CHECK-MESSAGES: :[[@LINE-2]]:9: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: } else {
  // CHECK-FIXES: }

  if (cond("if2") /*comment*/)
    // some comment
    do_something("if-multi-line");
  else {
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:31: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: if (cond("if2") /*comment*/) {
  // CHECK-FIXES: } else {

  if (cond("if2.1") /*comment*/) {
  } else
    // some comment
    do_something("else-multi-line");
  // CHECK-MESSAGES: :[[@LINE-3]]:9: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: } else {
  // CHECK-FIXES: }

  if (cond("if3")) do_something("elseif-same-line");
  else if (cond("if3")) {
  } else {
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:19: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: if (cond("if3")) { do_something("elseif-same-line");
  // CHECK-FIXES: } else if (cond("if3")) {

  if (cond("if3.1")) {
  } else if (cond("if3.1")) do_something("elseif-same-line");
  else {
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:28: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: } else if (cond("if3.1")) { do_something("elseif-same-line");
  // CHECK-FIXES: } else {

  if (cond("if3.2")) {
  } else if (cond("if3.2")) {
  } else do_something("else-same-line");
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: } else { do_something("else-same-line");
  // CHECK-FIXES: }

  if (cond("if4-outer"))
    if (cond("if4-inner"))
      do_something("if-single-line");
    else {
    }
  else {
  }
  // CHECK-MESSAGES: :[[@LINE-7]]:25: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-MESSAGES: :[[@LINE-7]]:27: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: if (cond("if4-outer")) {
  // CHECK-FIXES: if (cond("if4-inner")) {
  // CHECK-FIXES: } else {
  // CHECK-FIXES: } else {

  if (cond("if5"))
      do_something("if-single-line");
  else if (cond("if5")) {
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:19: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: if (cond("if5")) {
  // CHECK-FIXES: } else if (cond("if5")) {

  if (cond("if5.1")) {
  } else if (cond("if5.1"))
      do_something("elseif-single-line");
  // CHECK-MESSAGES: :[[@LINE-2]]:28: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: } else if (cond("if5.1")) {
  // CHECK-FIXES: }

  if (MACRO_COND("if6")) MACRO_FUN;
  else {
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:25: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: if (MACRO_COND("if6")) { MACRO_FUN;
  // CHECK-FIXES: } else {

  if (MACRO_COND("if6")) {
  } else MACRO_FUN;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: } else { MACRO_FUN;

  if (cond("if0")) goo:
    return;
  else too: {
    return;
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:19: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: if (cond("if0")) { goo:
  // CHECK-FIXES-NEXT:     return;
  // CHECK-FIXES-NEXT: } else too: {

  if (cond("if0")) xoo: {
    return;
  } else yoo:
    return;
  // CHECK-MESSAGES: :[[@LINE-2]]:9: warning: statement should have braces [readability-inconsistent-ifelse-braces]
  // CHECK-FIXES: } else { yoo:
  // CHECK-FIXES-NEXT:   return;
  // CHECK-FIXES-NEXT: }

}

// Negative tests.
void g() {
  if (cond("if0")) {
    do_something("if-single-line");
  } else if (cond("if0")) {
    do_something("elseif-single-line");
  } else {
    do_something("else-single-line");
  }

  if (cond("if1"))
    do_something("if-single-line");
  else if (cond("if1"))
    do_something("elseif-single-line");
  else
    do_something("else-single-line");

  if (cond("if2")) {
    return;
  } else foo: {
    return;
  }

  if (cond("if3")) {
    return;
  } else hoo: {
    return;
  }

  if (cond("if4")) joo: {
    return;
  } else {
    return;
  }

  if (cond("if5")) koo: {
    return;
  } else loo: {
    return;
  }

  if (cond("if6")) moo:
    return;
  else noo:
    return;

  if (cond("if7"))
    return;
  else ooo:
    return;
}
