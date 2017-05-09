// RUN: %check_clang_tidy %s readability-braces-around-statements %t

void do_something(const char *) {}

bool cond(const char *) {
  return false;
}

#define EMPTY_MACRO
#define EMPTY_MACRO_FUN()

void test() {
  if (cond("if0") /*comment*/) do_something("same-line");
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: statement should be inside braces
  // CHECK-FIXES:   if (cond("if0") /*comment*/) { do_something("same-line");
  // CHECK-FIXES: }

  if (cond("if0.1"))
    do_something("single-line");
  // CHECK-MESSAGES: :[[@LINE-2]]:21: warning: statement should be inside braces
  // CHECK-FIXES: if (cond("if0.1")) {
  // CHECK-FIXES: }

  if (cond("if1") /*comment*/)
    // some comment
    do_something("if1");
  // CHECK-MESSAGES: :[[@LINE-3]]:31: warning: statement should be inside braces
  // CHECK-FIXES: if (cond("if1") /*comment*/) {
  // CHECK-FIXES: }
  if (cond("if2")) {
    do_something("if2");
  }
  if (cond("if3"))
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:19: warning: statement should be inside braces
  // CHECK-FIXES: if (cond("if3")) {
  // CHECK-FIXES: }

  if (cond("if-else1"))
    do_something("if-else1");
  // CHECK-MESSAGES: :[[@LINE-2]]:24: warning: statement should be inside braces
  // CHECK-FIXES: if (cond("if-else1")) {
  // CHECK-FIXES: } else {
  else
    do_something("if-else1 else");
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: statement should be inside braces
  // CHECK-FIXES: }
  if (cond("if-else2")) {
    do_something("if-else2");
  } else {
    do_something("if-else2 else");
  }

  if (cond("if-else if-else1"))
    do_something("if");
  // CHECK-MESSAGES: :[[@LINE-2]]:32: warning: statement should be inside braces
  // CHECK-FIXES: } else if (cond("else if1")) {
  else if (cond("else if1"))
    do_something("else if");
  // CHECK-MESSAGES: :[[@LINE-2]]:29: warning: statement should be inside braces
  else
    do_something("else");
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: statement should be inside braces
  // CHECK-FIXES: }
  if (cond("if-else if-else2")) {
    do_something("if");
  } else if (cond("else if2")) {
    do_something("else if");
  } else {
    do_something("else");
  }

  for (;;)
    do_something("for");
  // CHECK-MESSAGES: :[[@LINE-2]]:11: warning: statement should be inside braces
  // CHECK-FIXES: for (;;) {
  // CHECK-FIXES: }
  for (;;) {
    do_something("for");
  }
  for (;;)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:11: warning: statement should be inside braces
  // CHECK-FIXES: for (;;) {
  // CHECK-FIXES: }

  int arr[4] = {1, 2, 3, 4};
  for (int a : arr)
    do_something("for-range");
  // CHECK-MESSAGES: :[[@LINE-2]]:20: warning: statement should be inside braces
  // CHECK-FIXES: for (int a : arr) {
  // CHECK-FIXES: }
  for (int a : arr) {
    do_something("for-range");
  }
  for (int a : arr)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:20: warning: statement should be inside braces
  // CHECK-FIXES: for (int a : arr) {
  // CHECK-FIXES: }

  while (cond("while1"))
    do_something("while");
  // CHECK-MESSAGES: :[[@LINE-2]]:25: warning: statement should be inside braces
  // CHECK-FIXES: while (cond("while1")) {
  // CHECK-FIXES: }
  while (cond("while2")) {
    do_something("while");
  }
  while (false)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:16: warning: statement should be inside braces
  // CHECK-FIXES: while (false) {
  // CHECK-FIXES: }

  do
    do_something("do1");
  while (cond("do1"));
  // CHECK-MESSAGES: :[[@LINE-3]]:5: warning: statement should be inside braces
  // CHECK-FIXES: do {
  // CHECK-FIXES: } while (cond("do1"));
  do {
    do_something("do2");
  } while (cond("do2"));

  do
    ;
  while (false);
  // CHECK-MESSAGES: :[[@LINE-3]]:5: warning: statement should be inside braces
  // CHECK-FIXES: do {
  // CHECK-FIXES: } while (false);

  if (cond("ifif1"))
    // comment
    if (cond("ifif2"))
      // comment
      /*comment*/ ; // comment
  // CHECK-MESSAGES: :[[@LINE-5]]:21: warning: statement should be inside braces
  // CHECK-MESSAGES: :[[@LINE-4]]:23: warning: statement should be inside braces
  // CHECK-FIXES: if (cond("ifif1")) {
  // CHECK-FIXES: if (cond("ifif2")) {
  // CHECK-FIXES: }
  // CHECK-FIXES-NEXT: }

  if (cond("ifif3"))
    // comment
    if (cond("ifif4")) {
      // comment
      /*comment*/; // comment
    }
  // CHECK-MESSAGES: :[[@LINE-6]]:21: warning: statement should be inside braces
  // CHECK-FIXES: if (cond("ifif3")) {
  // CHECK-FIXES: }

  if (cond("ifif5"))
    ; /* multi-line
        comment */
  // CHECK-MESSAGES: :[[@LINE-3]]:21: warning: statement should be inside braces
  // CHECK-FIXES: if (cond("ifif5")) {
  // CHECK-FIXES: }/* multi-line

  if (1) while (2) if (3) for (;;) do ; while(false) /**/;/**/
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: statement should be inside braces
  // CHECK-MESSAGES: :[[@LINE-2]]:19: warning: statement should be inside braces
  // CHECK-MESSAGES: :[[@LINE-3]]:26: warning: statement should be inside braces
  // CHECK-MESSAGES: :[[@LINE-4]]:35: warning: statement should be inside braces
  // CHECK-MESSAGES: :[[@LINE-5]]:38: warning: statement should be inside braces
  // CHECK-FIXES: if (1) { while (2) { if (3) { for (;;) { do { ; } while(false) /**/;/**/
  // CHECK-FIXES-NEXT: }
  // CHECK-FIXES-NEXT: }
  // CHECK-FIXES-NEXT: }
  // CHECK-FIXES-NEXT: }
}

void f(const char *p) {
  if (!p)
    f("\
");
  // CHECK-MESSAGES: :[[@LINE-3]]:10: warning: statement should be inside braces
  // CHECK-FIXES:      {{^  }}if (!p) {{{$}}
  // CHECK-FIXES-NEXT: {{^    }}f("\{{$}}
  // CHECK-FIXES-_NEXT: {{^}}");{{$}} FIXME: This breaks (http://llvm.org/PR26228)
  // CHECK-FIXES-_NEXT: {{^}}}{{$}}
}

#define M(x) x

int test_macros(bool b) {
  if (b) {
    return 1;
  } else
    M(return 2);
  // CHECK-MESSAGES: :[[@LINE-2]]:9: warning: statement should be inside braces
  // CHECK-FIXES: } else {
  // CHECK-FIXES-NEXT:   M(return 2);
  // CHECK-FIXES-NEXT: }
  M(
    for (;;)
      ;
  );
  // CHECK-MESSAGES: :[[@LINE-3]]:13: warning: statement should be inside braces
  // CHECK-FIXES: {{^}}    for (;;) {{{$}}
  // CHECK-FIXES-NEXT: {{^      ;$}}
  // CHECK-FIXES-NEXT: {{^}$}}
}
