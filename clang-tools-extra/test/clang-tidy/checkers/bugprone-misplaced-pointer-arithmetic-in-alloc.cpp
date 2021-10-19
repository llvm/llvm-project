// RUN: %check_clang_tidy %s bugprone-misplaced-pointer-arithmetic-in-alloc %t

class C {
  int num;
public:
  explicit C(int n) : num(n) {}
};

void bad_new(int n, int m) {
  C *p = new C(n) + 10;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: arithmetic operation is applied to the result of operator new() instead of its size-like argument
  // CHECK-FIXES: C *p = new C(n + 10);

  p = new C(n) - 10;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: arithmetic operation is applied to the result of operator new() instead of its size-like argument
  // CHECK-FIXES: p = new C(n - 10);

  p = new C(n) + m;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: arithmetic operation is applied to the result of operator new() instead of its size-like argument
  // CHECK-FIXES: p = new C(n + m);

  p = new C(n) - (m + 10);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: arithmetic operation is applied to the result of operator new() instead of its size-like argument
  // CHECK-FIXES: p = new C(n - (m + 10));

  p = new C(n) - m + 10;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: arithmetic operation is applied to the result of operator new() instead of its size-like argument
  // CHECK-FIXES: p = new C(n - m) + 10;
  // FIXME: Should be p = new C(n - m + 10);
}

void bad_new_array(int n, int m) {
  char *p = new char[n] + 10;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: arithmetic operation is applied to the result of operator new[]() instead of its size-like argument
  // CHECK-FIXES: char *p = new char[n + 10];

  p = new char[n] - 10;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: arithmetic operation is applied to the result of operator new[]() instead of its size-like argument
  // CHECK-FIXES: p = new char[n - 10];

  p = new char[n] + m;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: arithmetic operation is applied to the result of operator new[]() instead of its size-like argument
  // CHECK-FIXES: p = new char[n + m];

  p = new char[n] - (m + 10);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: arithmetic operation is applied to the result of operator new[]() instead of its size-like argument
  // CHECK-FIXES: p = new char[n - (m + 10)];

  p = new char[n] - m + 10;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: arithmetic operation is applied to the result of operator new[]() instead of its size-like argument
  // CHECK-FIXES: p = new char[n - m] + 10;
  // FIXME: should be p = new char[n - m + 10];
}

namespace std {
typedef decltype(sizeof(void*)) size_t;
}

void* operator new(std::size_t, void*);

void placement_new_ptr(void *buf, C *old) {
  C **p = new (buf) C*(old) + 1;
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:11: warning: arithmetic operation is applied to the result of operator new() instead of its size-like argument
}
