// RUN: %check_clang_tidy %s readability-else-after-return %t -- -- -std=c++11 -fexceptions

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

extern int *g();
extern void h(int **x);

int *decl_in_condition() {
  if (int *x = g()) {
    return x;
  } else {
    h(&x);
    return x;
  }
}
