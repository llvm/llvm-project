// RUN: %check_clang_tidy %s readability-else-after-return %t -- -- -fexceptions -std=c++17

namespace std {
struct string {
  string(const char *);
  ~string();
};
} // namespace std

struct my_exception {
  my_exception(const std::string &s);
};

void f(int a) {
  if (a > 0)
    return;
  else // comment-0
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not use 'else' after 'return'
  // CHECK-FIXES: {{^}}  // comment-0
    return;

  if (a > 0) {
    return;
  } else { // comment-1
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: do not use 'else' after 'return'
  // CHECK-FIXES: {{^}}  } // comment-1
    return;
  }

  if (a > 0) {
    f(0);
    if (a > 10)
      return;
  } else {
    return;
  }

  if (a > 0)
    f(0);
  else if (a > 10)
    return;
  else // comment-2
  // CHECK-FIXES-NOT: {{^}}  // comment-2
    f(0);

  if (a > 0)
    if (a < 10)
      return;
    else // comment-3
    // CHECK-FIXES-NOT: {{^}}    // comment-3
      f(0);
  else
    if (a > 10)
      return;
    else // comment-4
    // CHECK-FIXES-NOT: {{^}}    // comment-4
      f(0);

  if (a > 0) {
    if (a < 10)
      return;
    else // comment-5
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: do not use 'else' after 'return'
    // CHECK-FIXES: {{^}}    // comment-5
      f(0);
  } else {
    if (a > 10)
      return;
    else // comment-6
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: do not use 'else' after 'return'
    // CHECK-FIXES: {{^}}    // comment-6
      f(0);
  }
}

void foo() {
  for (unsigned x = 0; x < 42; ++x) {
    if (x) {
      continue;
    } else { // comment-7
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: do not use 'else' after 'continue'
    // CHECK-FIXES: {{^}}    } // comment-7
      x++;
    }
    if (x) {
      break;
    } else { // comment-8
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: do not use 'else' after 'break'
    // CHECK-FIXES: {{^}}    } // comment-8
      x++;
    }
    if (x) {
      throw 42;
    } else { // comment-9
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: do not use 'else' after 'throw'
    // CHECK-FIXES: {{^}}    } // comment-9
      x++;
    }
    if (x) {
      throw my_exception("foo");
    } else { // comment-10
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: do not use 'else' after 'throw'
    // CHECK-FIXES: {{^}}    } // comment-10
      x++;
    }
  }
}

int g();
int h(int);

int declInConditionUsedInElse() {
  if (int X = g()) { // comment-11
    // CHECK-FIXES: {{^}}  int X = g();
    // CHECK-FIXES-NEXT: {{^}}if (X) { // comment-11
    return X;
  } else { // comment-11
           // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: do not use 'else' after 'return'
           // CHECK-FIXES: {{^}}  } // comment-11
    return h(X);
  }
}
int declInConditionUnusedInElse() {
  if (int X = g()) {
    return h(X);
  } else { // comment-12
           // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: do not use 'else' after 'return'
           // CHECK-FIXES: {{^}}  } // comment-12
    return 0;
  }
}

int varInitAndCondition() {
  if (int X = g(); X != 0) { // comment-13
    // CHECK-FIXES: {{^}}  int X = g();
    // CHECK-FIXES-NEXT: {{^}}if ( X != 0) { // comment-13
    return X;
  } else { // comment-13
           // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: do not use 'else' after 'return'
           // CHECK-FIXES: {{^}}  } // comment-13
    return h(X);
  }
}

int varInitAndConditionUnusedInElse() {
  if (int X = g(); X != 0) {
    return X;
  } else { // comment-14
           // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: do not use 'else' after 'return'
           // CHECK-FIXES: {{^}}  } // comment-14
    return 0;
  }
}

int initAndCondition() {
  int X;
  if (X = g(); X != 0) {
    return X;
  } else { // comment-15
           // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: do not use 'else' after 'return'
           // CHECK-FIXES: {{^}}  } // comment-15
    return h(X);
  }
}

int varInitAndConditionUnusedInElseWithDecl() {
  int Y = g();
  if (int X = g(); X != 0) {
    return X;
  } else { // comment-16
           // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: do not use 'else' after 'return'
           // CHECK-FIXES-NOT: {{^}}  } //comment-16
    int Y = g();
    h(Y);
  }
  return Y;
}

int varInitAndCondVarUsedInElse() {
  if (int X = g(); int Y = g()) { // comment-17
    // CHECK-FIXES:      {{^}}  int X = g();
    // CHECK-FIXES-NEXT: {{^}}int Y = g();
    // CHECK-FIXES-NEXT: {{^}}if ( Y) { // comment-17
    return X ? X : Y;
  } else { // comment-17
           // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: do not use 'else' after 'return'
           // CHECK-FIXES: {{^}}  } // comment-17
    return X ? X : h(Y);
  }
}

int lifeTimeExtensionTests(int a) {
  if (a > 0) {
    return a;
  } else {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: do not use 'else' after 'return'
    int b = 0;
    h(b);
  }
  if (int b = a; (b & 1) == 0) {
    return a;
  } else {
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: do not use 'else' after 'return'
    b++;
  }
  if (int b = a; b > 1) { // comment-18
    // CHECK-FIXES:      {{^}}  int b = a;
    // CHECK-FIXES-NEXT: {{^}}if ( b > 1) { // comment-18
    return a;
  } else { // comment-18
           // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: do not use 'else' after 'return'
           // CHECK-FIXES: {{^}}  } // comment-18
    return b;
  }
}

void test_B44745() {
  // This is the actual minimum test case for the crash in bug 44745. We aren't
  // too worried about the warning or fix here, more we don't want a crash.
  // CHECK-MESSAGES: :[[@LINE+3]]:5: warning: do not use 'else' after 'return' [readability-else-after-return]
  if (auto X = false) {
    return;
  } else {
    for (;;) {
    }
  }
  return;
}

void testPPConditionals() {

  // These cases the return isn't inside the conditional so diagnose as normal.
  if (true) {
    return;
#if 1
#endif
  } else {
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: do not use 'else' after 'return'
    return;
  }
  if (true) {
#if 1
#endif
    return;
  } else {
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: do not use 'else' after 'return'
    return;
  }

  // No return here in the AST, no special handling needed.
  if (true) {
#if 0
    return;
#endif
  } else {
    return;
  }

  // Return here is inside a preprocessor conditional block, ignore this case.
  if (true) {
#if 1
    return;
#endif
  } else {
    return;
  }

  // These cases, same as above but with an #else block.
  if (true) {
#if 1
    return;
#else
#endif
  } else {
    return;
  }
  if (true) {
#if 0
#else
    return;
#endif
  } else {
    return;
  }

// Ensure it can handle macros.
#define RETURN return
  if (true) {
#if 1
    RETURN;
#endif
  } else {
    return;
  }
#define ELSE else
  if (true) {
#if 1
    return;
#endif
  }
  ELSE {
    return;
  }

  // Whole statement is in a conditional block so diagnose as normal.
#if 1
  if (true) {
    return;
  } else {
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: do not use 'else' after 'return'
    return;
  }
#endif
}
