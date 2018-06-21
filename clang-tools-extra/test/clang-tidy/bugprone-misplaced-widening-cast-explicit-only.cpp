// RUN: %check_clang_tidy %s bugprone-misplaced-widening-cast %t -- -config="{CheckOptions: [{key: bugprone-misplaced-widening-cast.CheckImplicitCasts, value: 0}]}" --

void func(long arg) {}

void assign(int a, int b) {
  long l;

  l = a * b;
  l = (long)(a * b);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: either cast from 'int' to 'long' is ineffective, or there is loss of precision before the conversion [bugprone-misplaced-widening-cast]
  l = (long)a * b;

  l = a << 8;
  l = (long)(a << 8);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: either cast from 'int' to 'long'
  l = (long)b << 8;

  l = static_cast<long>(a * b);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: either cast from 'int' to 'long'
}

void compare(int a, int b, long c) {
  bool l;

  l = a * b == c;
  l = c == a * b;
  l = (long)(a * b) == c;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: either cast from 'int' to 'long'
  l = c == (long)(a * b);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: either cast from 'int' to 'long'
  l = (long)a * b == c;
  l = c == (long)a * b;
}

void init(unsigned int n) {
  long l1 = n << 8;
  long l2 = (long)(n << 8);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: either cast from 'unsigned int' to 'long'
  long l3 = (long)n << 8;
}

void call(unsigned int n) {
  func(n << 8);
  func((long)(n << 8));
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: either cast from 'unsigned int' to 'long'
  func((long)n << 8);
}

long ret(int a) {
  if (a < 0) {
    return a * 1000;
  } else if (a > 0) {
    return (long)(a * 1000);
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: either cast from 'int' to 'long'
  } else {
    return (long)a * 1000;
  }
}

// Shall not generate an assert. https://bugs.llvm.org/show_bug.cgi?id=33660
template <class> class A {
  enum Type {};
  static char *m_fn1() { char p = (Type)(&p - m_fn1()); }
};
