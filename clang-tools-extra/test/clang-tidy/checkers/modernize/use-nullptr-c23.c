// RUN: %check_clang_tidy -std=c23-or-later %s modernize-use-nullptr %t

#define NULL 0

void test_assignment() {
  int *p1 = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use nullptr [modernize-use-nullptr]
  // CHECK-FIXES: int *p1 = nullptr;
  p1 = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use nullptr
  // CHECK-FIXES: p1 = nullptr;

  int *p2 = NULL;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use nullptr
  // CHECK-FIXES: int *p2 = nullptr;

  p2 = p1;
  // CHECK-FIXES: p2 = p1;

  const int null = 0;
  int *p3 = &null;

  p3 = NULL;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use nullptr
  // CHECK-FIXES: p3 = nullptr;

  int *p4 = p3;

  int i1 = 0;

  int i2 = NULL;

  int i3 = null;

  int *p5, *p6, *p7;
  p5 = p6 = p7 = NULL;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: use nullptr
  // CHECK-FIXES: p5 = p6 = p7 = nullptr;
}

void test_function(int *p) {}

void test_function_no_ptr_param(int i) {}

void test_function_call() {
  test_function(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use nullptr
  // CHECK-FIXES: test_function(nullptr);

  test_function(NULL);
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use nullptr
  // CHECK-FIXES: test_function(nullptr);

  test_function_no_ptr_param(0);
}

char *test_function_return1() {
  return 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use nullptr
  // CHECK-FIXES: return nullptr;
}

void *test_function_return2() {
  return NULL;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use nullptr
  // CHECK-FIXES: return nullptr;
}

int test_function_return4() {
  return 0;
}

int test_function_return5() {
  return NULL;
}

int *test_function_return_cast1() {
  return(int)0;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use nullptr
  // CHECK-FIXES: return nullptr;
}

int *test_function_return_cast2() {
#define RET return
  RET(int)0;
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use nullptr
  // CHECK-FIXES: RET nullptr;
#undef RET
}

// Test parentheses expressions resulting in a nullptr.
int *test_parentheses_expression1() {
  return(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use nullptr
  // CHECK-FIXES: return(nullptr);
}

int *test_parentheses_expression2() {
  return((int)(0.0f));
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use nullptr
  // CHECK-FIXES: return(nullptr);
}

int *test_nested_parentheses_expression() {
  return((((0))));
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use nullptr
  // CHECK-FIXES: return((((nullptr))));
}

void test_const_pointers() {
  const int *const_p1 = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: use nullptr
  // CHECK-FIXES: const int *const_p1 = nullptr;
  const int *const_p2 = NULL;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: use nullptr
  // CHECK-FIXES: const int *const_p2 = nullptr;
  const int *const_p3 = (int)0;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: use nullptr
  // CHECK-FIXES: const int *const_p3 = nullptr;
  const int *const_p4 = (int)0.0f;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: use nullptr
  // CHECK-FIXES: const int *const_p4 = nullptr;
}

void test_nested_implicit_cast_expr() {
  int func0(void*, void*);
  int func1(int, void*, void*);

  (double)func1(0, 0, 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use nullptr
  // CHECK-MESSAGES: :[[@LINE-2]]:23: warning: use nullptr
  // CHECK-FIXES: (double)func1(0, nullptr, nullptr);
  (double)func1(func0(0, 0), 0, 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: use nullptr
  // CHECK-MESSAGES: :[[@LINE-2]]:26: warning: use nullptr
  // CHECK-MESSAGES: :[[@LINE-3]]:30: warning: use nullptr
  // CHECK-MESSAGES: :[[@LINE-4]]:33: warning: use nullptr
  // CHECK-FIXES: (double)func1(func0(nullptr, nullptr), nullptr, nullptr);
}
