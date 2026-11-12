// RUN: %check_clang_tidy %s readability-redundant-function-ptr-dereference %t

void f(int i);

void positive() {
  void (*p)(int) = f;

  (**p)(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:4: warning: redundant repeated dereference of function pointer [readability-redundant-function-ptr-dereference]
  // CHECK-FIXES: (*p)(1);
  (*****p)(2);
  // CHECK-MESSAGES: :[[@LINE-1]]:4: warning: redundant repeated
  // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: redundant repeated
  // CHECK-MESSAGES: :[[@LINE-3]]:6: warning: redundant repeated
  // CHECK-MESSAGES: :[[@LINE-4]]:7: warning: redundant repeated
  // CHECK-FIXES: (*p)(2);
}

template<typename T>
void invoke(const T& fn) {
  fn(0); // 1
  (*fn)(0); // 2
  // CHECK-MESSAGES: :[[@LINE-1]]:4: warning: redundant repeated
  // CHECK-FIXES: fn(0); // 1
  // CHECK-FIXES: (fn)(0); // 2
  // FIXME: Remove unnecessary parentheses.
}

void f1(int);
void f2(double);
void f3(char);

void instantiate() {
  invoke(f1);
  invoke(f2);
  invoke(f3);
  invoke([](unsigned) {});
}

void negative() {
  void (*q)(int) = &f;

  q(1);
  (*q)(2);
}
