// RUN: %check_clang_tidy %s bugprone-misplaced-widening-cast %t -- -config="{CheckOptions: [{key: bugprone-misplaced-widening-cast.CheckImplicitCasts, value: 1}]}" --

void func(long arg) {}

void assign(int a, int b) {
  long l;

  l = a * b;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: either cast from 'int' to 'long' is ineffective, or there is loss of precision before the conversion [bugprone-misplaced-widening-cast]
  l = (long)(a * b);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: either cast from 'int' to 'long'
  l = (long)a * b;

  l = a << 8;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: either cast from 'int' to 'long'
  l = (long)(a << 8);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: either cast from 'int' to 'long'
  l = (long)b << 8;

  l = static_cast<long>(a * b);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: either cast from 'int' to 'long'
}

void compare(int a, int b, long c) {
  bool l;

  l = a * b == c;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: either cast from 'int' to 'long'
  l = c == a * b;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: either cast from 'int' to 'long'
  l = (long)(a * b) == c;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: either cast from 'int' to 'long'
  l = c == (long)(a * b);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: either cast from 'int' to 'long'
  l = (long)a * b == c;
  l = c == (long)a * b;
}

void init(unsigned int n) {
  long l1 = n << 8;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: either cast from 'unsigned int' to 'long'
  long l2 = (long)(n << 8);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: either cast from 'unsigned int' to 'long'
  long l3 = (long)n << 8;
}

void call(unsigned int n) {
  func(n << 8);
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: either cast from 'unsigned int' to 'long'
  func((long)(n << 8));
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: either cast from 'unsigned int' to 'long'
  func((long)n << 8);
}

long ret(int a) {
  if (a < 0) {
    return a * 1000;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: either cast from 'int' to 'long'
  } else if (a > 0) {
    return (long)(a * 1000);
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: either cast from 'int' to 'long'
  } else {
    return (long)a * 1000;
  }
}

void dontwarn1(unsigned char a, int i, unsigned char *p) {
  long l;
  // The result is a 9 bit value, there is no truncation in the implicit cast.
  l = (long)(a + 15);
  // The result is a 12 bit value, there is no truncation in the implicit cast.
  l = (long)(a << 4);
  // The result is a 3 bit value, there is no truncation in the implicit cast.
  l = (long)((i % 5) + 1);
  // The result is a 16 bit value, there is no truncation in the implicit cast.
  l = (long)(((*p) << 8) + *(p + 1));
}

template <class T> struct DontWarn2 {
  void assign(T a, T b) {
    long l;
    l = (long)(a * b);
  }
};
DontWarn2<int> DW2;

// Cast is not suspicious when casting macro.
#define A  (X<<2)
long macro1(int X) {
  return (long)A;
}

// Don't warn about cast in macro.
#define B(X,Y)   (long)(X*Y)
long macro2(int x, int y) {
  return B(x,y);
}

void floatingpoint(float a, float b) {
  double d = (double)(a * b); // Currently we don't warn for this.
}
