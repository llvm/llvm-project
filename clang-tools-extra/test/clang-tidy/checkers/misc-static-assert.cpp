// RUN: %check_clang_tidy %s misc-static-assert %t

void abort() {}
#ifdef NDEBUG
#define assert(x) 1
#else
#define assert(x)                                                              \
  if (!(x))                                                                    \
  abort()
#endif

void print(...);

#define ZERO_MACRO 0

#define False false
#define FALSE 0

#define my_macro() assert(0 == 1)
// CHECK-FIXES: #define my_macro() assert(0 == 1)

constexpr bool myfunc(int a, int b) { return a * b == 0; }

typedef __SIZE_TYPE__ size_t;
extern "C" size_t strlen(const char *s);

class A {
public:
  bool method() { return true; }
};

class B {
public:
  constexpr bool method() { return true; }
};

template <class T> void doSomething(T t) {
  assert(myfunc(1, 2));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: found assert() that could be replaced by static_assert() [misc-static-assert]
  // CHECK-FIXES: {{^  }}static_assert(myfunc(1, 2), "");

  assert(t.method());
  // CHECK-FIXES: {{^  }}assert(t.method());

  assert(sizeof(T) == 123);
}

int main() {
  my_macro();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: found assert() that could be
  // CHECK-FIXES: {{^  }}my_macro();

  assert(myfunc(1, 2) && (3 == 4));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: found assert() that could be
  // CHECK-FIXES: {{^  }}static_assert(myfunc(1, 2) && (3 == 4), "");

  int x = 1;
  assert(x == 0);
  // CHECK-FIXES: {{^  }}assert(x == 0);

  A a;
  B b;

  doSomething<A>(a);
  doSomething<B>(b);

  assert(false);
  // CHECK-FIXES: {{^  }}assert(false);

  assert(False);
  // CHECK-FIXES: {{^  }}assert(False);
  assert(FALSE);
  // CHECK-FIXES: {{^  }}assert(FALSE);

  assert(ZERO_MACRO);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: found assert() that could be
  // CHECK-FIXES: {{^  }}static_assert(ZERO_MACRO, "");

  assert(!"Don't report me!");
  // CHECK-FIXES: {{^  }}assert(!"Don't report me!");

  assert(0 && "Don't report me!");
  // CHECK-FIXES: {{^  }}assert(0 && "Don't report me!");

  assert(false && "Don't report me!");
  // CHECK-FIXES: {{^  }}assert(false && "Don't report me!");

#define NULL ((void*)0)
  assert(NULL && "Don't report me!");
  // CHECK-FIXES: {{^  }}assert(NULL && "Don't report me!");

  assert(NULL == "Don't report me!");
  // CHECK-FIXES: {{^  }}assert(NULL == "Don't report me!");

  assert("Don't report me!" == NULL);
  // CHECK-FIXES: {{^  }}assert("Don't report me!" == NULL);

  assert(0 == "Don't report me!");
  // CHECK-FIXES: {{^  }}assert(0 == "Don't report me!");

#define NULL ((unsigned int)0)
  assert(NULL && "Report me!");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: found assert() that could be
  // CHECK-FIXES: {{^  }}static_assert(NULL , "Report me!");

#define NULL __null
  assert(__null == "Don't report me!");
  // CHECK-FIXES: {{^  }}assert(__null == "Don't report me!");
  assert(NULL == "Don't report me!");
  // CHECK-FIXES: {{^  }}assert(NULL == "Don't report me!");
#undef NULL

  assert(ZERO_MACRO && "Report me!");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: found assert() that could be
  // CHECK-FIXES: {{^  }}static_assert(ZERO_MACRO , "Report me!");

  assert(0);

#define false false
  assert(false);

#define false 0
  assert(false);
#undef false

  assert(10==5 && "Report me!");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: found assert() that could be
  // CHECK-FIXES: {{^  }}static_assert(10==5 , "Report me!");

  assert(strlen("12345") == 5);
  // CHECK-FIXES: {{^  }}assert(strlen("12345") == 5);

#define assert(e) (__builtin_expect(!(e), 0) ? print (#e, __FILE__, __LINE__) : (void)0)
  assert(false);
  // CHECK-FIXES: {{^  }}assert(false);

  assert(10 == 5 + 5);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: found assert() that could be
  // CHECK-FIXES: {{^  }}static_assert(10 == 5 + 5, "");
#undef assert

  return 0;
}
