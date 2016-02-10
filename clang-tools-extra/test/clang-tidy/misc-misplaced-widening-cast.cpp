// RUN: %check_clang_tidy %s misc-misplaced-widening-cast %t -- -- -target x86_64-unknown-unknown

void assign(int a, int b) {
  long l;

  l = a * b;
  l = (long)(a * b);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: either cast from 'int' to 'long' is ineffective, or there is loss of precision before the conversion [misc-misplaced-widening-cast]
  l = (long)a * b;

  l = (long)(a << 8);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: either cast from 'int' to 'long'
  l = (long)b << 8;

  l = static_cast<long>(a * b);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: either cast from 'int' to 'long' is ineffective, or there is loss of precision before the conversion [misc-misplaced-widening-cast]
}

void init(unsigned int n) {
  long l = (long)(n << 8);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: either cast from 'unsigned int'
}

long ret(int a) {
  return (long)(a * 1000);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: either cast from 'int' to 'long'
}

void dontwarn1(unsigned char a, int i, unsigned char *p) {
  long l;
  // The result is a 9 bit value, there is no truncation in the implicit cast.
  l = (long)(a + 15);
  // The result is a 12 bit value, there is no truncation in the implicit cast.
  l = (long)(a << 4);
  // The result is a 3 bit value, there is no truncation in the implicit cast.
  l = (long)((i%5)+1);
  // The result is a 16 bit value, there is no truncation in the implicit cast.
  l = (long)(((*p)<<8) + *(p+1));
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
