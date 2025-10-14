// RUN: %check_clang_tidy %s modernize-use-nullptr %t -- -- -fno-delayed-template-parsing

const unsigned int g_null = 0;
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

  int *p3;
  p3 = NULL;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use nullptr
  // CHECK-FIXES: p3 = nullptr;

  int *p4 = p3;
  // CHECK-FIXES: int *p4 = p3;

  int i1 = 0;

  int i2 = NULL;

  const int null = 0;
  int i3 = null;

  int *p5, *p6, *p7;
  p5 = p6 = p7 = NULL;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: use nullptr
  // CHECK-FIXES: p5 = p6 = p7 = nullptr;
}

struct Foo {
  Foo(int *p = NULL) : m_p1(p) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use nullptr
  // CHECK-FIXES: Foo(int *p = nullptr) : m_p1(p) {}

  void bar(int *p = 0) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use nullptr
  // CHECK-FIXES: void bar(int *p = nullptr) {}

  void baz(int i = 0) {}

  int *m_p1;
  static int *m_p2;
};

int *Foo::m_p2 = NULL;
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: use nullptr
// CHECK-FIXES: int *Foo::m_p2 = nullptr;

struct Baz {
  Baz() : i(0) {}
  int i;
};

void test_cxx_cases() {
  Foo f;

  f.bar(NULL);
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use nullptr
  // CHECK-FIXES: f.bar(nullptr);

  f.baz(g_null);

  f.m_p1 = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use nullptr
  // CHECK-FIXES: f.m_p1 = nullptr;

  Baz b2;
  int Baz::*memptr(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use nullptr
  // CHECK-FIXES: int Baz::*memptr(nullptr);

  memptr = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use nullptr
  // CHECK-FIXES: memptr = nullptr;
}

void test_function_default_param1(void *p = 0);
// CHECK-MESSAGES: :[[@LINE-1]]:45: warning: use nullptr
// CHECK-FIXES: void test_function_default_param1(void *p = nullptr);

void test_function_default_param2(void *p = NULL);
// CHECK-MESSAGES: :[[@LINE-1]]:45: warning: use nullptr
// CHECK-FIXES: void test_function_default_param2(void *p = nullptr);

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

int test_function_return3() {
  return 0;
}

int test_function_return4() {
  return NULL;
}

int test_function_return5() {
  return g_null;
}

int *test_function_return_cast() {
#define RET return
  RET 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use nullptr
  // CHECK-FIXES: RET nullptr;
#undef RET
}

// Test parentheses expressions resulting in a nullptr.
int *test_parentheses_expression() {
  return(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use nullptr
  // CHECK-FIXES: return(nullptr);
}

int *test_nested_parentheses_expression() {
  return((((0))));
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use nullptr
  // CHECK-FIXES: return((((nullptr))));
}

void *test_parentheses_explicit_cast() {
  return(static_cast<void*>(0));
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: use nullptr
  // CHECK-FIXES: return(static_cast<void*>(nullptr));
}

void *test_parentheses_explicit_cast_sequence1() {
  return(static_cast<void*>(static_cast<int*>((void*)NULL)));
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: use nullptr
  // CHECK-FIXES: return(static_cast<void*>(nullptr));
}

void *test_parentheses_explicit_cast_sequence2() {
  return(static_cast<void*>(reinterpret_cast<int*>((float*)(0))));
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: use nullptr
  // CHECK-FIXES: return(static_cast<void*>(nullptr));
}

// Test explicit cast expressions resulting in nullptr.
struct Bam {
  Bam(int *a) {}
  Bam(float *a) {}
  Bam operator=(int *a) { return Bam(a); }
  Bam operator=(float *a) { return Bam(a); }
};

void ambiguous_function(int *a) {}
void ambiguous_function(float *a) {}
void const_ambiguous_function(const int *p) {}
void const_ambiguous_function(const float *p) {}

void test_explicit_cast_ambiguous1() {
  ambiguous_function((int*)0);
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use nullptr
  // CHECK-FIXES: ambiguous_function((int*)nullptr);
}

void test_explicit_cast_ambiguous2() {
  ambiguous_function((int*)(0));
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use nullptr
  // CHECK-FIXES: ambiguous_function((int*)nullptr);
}

void test_explicit_cast_ambiguous3() {
  ambiguous_function(static_cast<int*>(reinterpret_cast<int*>((float*)0)));
  // CHECK-MESSAGES: :[[@LINE-1]]:40: warning: use nullptr
  // CHECK-FIXES: ambiguous_function(static_cast<int*>(nullptr));
}

Bam test_explicit_cast_ambiguous4() {
  return(((int*)(0)));
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use nullptr
  // CHECK-FIXES: return(((int*)nullptr));
}

void test_explicit_cast_ambiguous5() {
  // Test for ambiguous overloaded constructors.
  Bam k((int*)(0));
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use nullptr
  // CHECK-FIXES: Bam k((int*)nullptr);

  // Test for ambiguous overloaded operators.
  k = (int*)0;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use nullptr
  // CHECK-FIXES: k = (int*)nullptr;
}

void test_const_pointers_abiguous() {
  const_ambiguous_function((int*)0);
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: use nullptr
  // CHECK-FIXES: const_ambiguous_function((int*)nullptr);
}

// Test where the implicit cast to null is surrounded by another implicit cast
// with possible explicit casts in-between.
void test_const_pointers() {
  const int *const_p1 = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: use nullptr
  // CHECK-FIXES: const int *const_p1 = nullptr;
  const int *const_p2 = NULL;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: use nullptr
  // CHECK-FIXES: const int *const_p2 = nullptr;
  const int *const_p3 = (int*)0;
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: use nullptr
  // CHECK-FIXES: const int *const_p3 = (int*)nullptr;
  int *t;
  const int *const_p4 = static_cast<int*>(t ? t : static_cast<int*>(0));
  // CHECK-MESSAGES: :[[@LINE-1]]:69: warning: use nullptr
  // CHECK-FIXES: const int *const_p4 = static_cast<int*>(t ? t : static_cast<int*>(nullptr));
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

// FIXME: currently, the check doesn't work as it should with templates.
template<typename T>
class A {
 public:
  A(T *p = NULL) {
    Ptr = static_cast<T*>(NULL);

    Ptr = static_cast<T*>(reinterpret_cast<int*>((void*)NULL));
    // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: use nullptr
    // CHECK-FIXES: Ptr = static_cast<T*>(nullptr);
    // FIXME: a better fix-it is: Ptr = nullptr;

    T *p2 = static_cast<T*>(reinterpret_cast<int*>((void*)NULL));
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: use nullptr
    // CHECK-FIXES: T *p2 = static_cast<T*>(nullptr);
    // FIXME: a better fix-it is: T *p2 = nullptr;

    Ptr = NULL;
  }

  void f() {
    Ptr = NULL;
  }
  T *Ptr;
};

template<typename T>
T *f2(T *a = NULL) {
  return a ? a : NULL;
}
