// RUN: %check_clang_tidy %s readability-redundant-control-flow %t

void g(int i);
void j();

void f() {
  return;
}
// CHECK-MESSAGES: :[[@LINE-2]]:3: warning: redundant return statement at the end of a function with a void return type [readability-redundant-control-flow]
// CHECK-FIXES: void f() {
// CHECK-FIXES-NEXT: {{^ *}$}}

void g() {
  f();
  return;
}
// CHECK-MESSAGES: :[[@LINE-2]]:3: warning: redundant return statement
// CHECK-FIXES: f();
// CHECK-FIXES-NEXT: {{^ *}$}}

void g(int i) {
  if (i < 0) {
    return;
  }
  if (i < 10) {
    f();
  }
}

int h() {
  return 1;
}

void j() {
}

void k() {
  for (int i = 0; i < 10; ++i) {
    continue;
  }
}
// CHECK-MESSAGES: :[[@LINE-3]]:5: warning: redundant continue statement at the end of loop statement
// CHECK-FIXES: for (int i = 0; i < 10; ++i) {
// CHECK-FIXES-NEXT: {{^ *}$}}

void k2() {
  int v[10] = { 0 };
  for (auto i : v) {
    continue;
  }
}
// CHECK-MESSAGES: :[[@LINE-3]]:5: warning: redundant continue statement
// CHECK-FIXES: for (auto i : v) {
// CHECK-FIXES-NEXT: {{^ *}$}}

void m() {
  int i = 0;
  do {
    ++i;
    continue;
  } while (i < 10);
}
// CHECK-MESSAGES: :[[@LINE-3]]:5: warning: redundant continue statement
// CHECK-FIXES: {{^  do {$}}
// CHECK-FIXES-NEXT: ++i;
// CHECK-FIXES-NEXT: } while (i < 10);

void p() {
  int i = 0;
  while (i < 10) {
    ++i;
    continue;
  }
}
// CHECK-MESSAGES: :[[@LINE-3]]:5: warning: redundant continue statement
// CHECK-FIXES: while (i < 10) {
// CHECK-FIXES-NEXT: ++i;
// CHECK-FIXES-NEXT: {{^ *}$}}

void im_not_dead(int i) {
  if (i > 0) {
    return;
  }
  g();
}

void im_still_not_dead(int i) {
  for (int j = 0; j < 10; ++j) {
    if (i < 10) {
      continue;
    }
    g();
  }
}

void im_dead(int i) {
  if (i > 0) {
    return;
    g();
  }
  g();
}

void im_still_dead(int i) {
  for (int j = 0; j < 10; ++j) {
    if (i < 10) {
      continue;
      g();
    }
    g();
  }
}

void void_return() {
  return g();
}

void nested_return_unmolested() {
  g();
  {
    g();
    return;
  }
}

void nested_continue_unmolested() {
  for (int i = 0; i < 10; ++i) {
    if (i < 5) {
      continue;
    }
  }
}

#define MACRO_RETURN_UNMOLESTED(fn_)  \
  (fn_)();                            \
  return

#define MACRO_CONTINUE_UNMOLESTED(x_) \
  do {                                \
    for (int i = 0; i < (x_); ++i) {  \
      continue;                       \
    }                                 \
  } while (false)

void macro_return() {
  MACRO_RETURN_UNMOLESTED(g);
}

void macro_continue() {
  MACRO_CONTINUE_UNMOLESTED(10);
}

#define MACRO_RETURN_ARG(stmt_) \
  stmt_

#define MACRO_CONTINUE_ARG(stmt_)   \
  do {                              \
    for (int i = 0; i < 10; ++i) {  \
      stmt_;                        \
    }                               \
  } while (false)

void macro_arg_return() {
  MACRO_RETURN_ARG(return);
}

void macro_arg_continue() {
  MACRO_CONTINUE_ARG(continue);
}

template <typename T>
void template_return(T check) {
  if (check < T(0)) {
    return;
  }
  return;
}
// CHECK-MESSAGES: :[[@LINE-2]]:3: warning: redundant return statement
// CHECK-FIXES: if (check < T(0)) {
// CHECK-FIXES-NEXT: {{^    return;$}}
// CHECK-FIXES-NEXT: {{^ *}$}}
// CHECK-FIXES-NEXT: {{^ *}$}}

template <>
void template_return(int check) {
  if (check < 0) {
    return;
  }
  return;
}
// CHECK-MESSAGES: :[[@LINE-2]]:3: warning: redundant return statement
// CHECK-FIXES: if (check < 0) {
// CHECK-FIXES-NEXT: {{^    return;$}}
// CHECK-FIXES-NEXT: {{^ *}$}}
// CHECK-FIXES-NEXT: {{^ *}$}}

template <typename T>
void template_loop(T end) {
  for (T i = 0; i < end; ++i) {
    continue;
  }
}
// CHECK-MESSAGES: :[[@LINE-3]]:5: warning: redundant continue statement
// CHECK-FIXES: for (T i = 0; i < end; ++i) {
// CHECK-FIXES-NEXT: {{^ *}$}}

template <>
void template_loop(int end) {
  for (int i = 0; i < end; ++i) {
    continue;
  }
}
// CHECK-MESSAGES: :[[@LINE-3]]:5: warning: redundant continue statement
// CHECK-FIXES: for (int i = 0; i < end; ++i) {
// CHECK-FIXES-NEXT: {{^ *}$}}

void call_templates() {
  template_return(10);
  template_return(10.0f);
  template_return(10.0);
  template_loop(10);
  template_loop(10L);
  template_loop(10U);
}
