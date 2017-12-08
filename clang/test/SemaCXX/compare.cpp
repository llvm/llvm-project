// Force x86-64 because some of our heuristics are actually based
// on integer sizes.

// RUN: %clang_cc1 -triple x86_64-apple-darwin -fsyntax-only -pedantic -verify -Wsign-compare -std=c++11 %s

int test0(long a, unsigned long b) {
  enum EnumA {A};
  enum EnumB {B};
  enum EnumC {C = 0x10000};
  return
         // (a,b)
         (a == (unsigned long) b) +  // expected-warning {{comparison of integers of different signs}}
         (a == (unsigned int) b) +
         (a == (unsigned short) b) +
         (a == (unsigned char) b) +
         ((long) a == b) +  // expected-warning {{comparison of integers of different signs}}
         ((int) a == b) +  // expected-warning {{comparison of integers of different signs}}
         ((short) a == b) +  // expected-warning {{comparison of integers of different signs}}
         ((signed char) a == b) +  // expected-warning {{comparison of integers of different signs}}
         ((long) a == (unsigned long) b) +  // expected-warning {{comparison of integers of different signs}}
         ((int) a == (unsigned int) b) +  // expected-warning {{comparison of integers of different signs}}
         ((short) a == (unsigned short) b) +
         ((signed char) a == (unsigned char) b) +
         (a < (unsigned long) b) +  // expected-warning {{comparison of integers of different signs}}
         (a < (unsigned int) b) +
         (a < (unsigned short) b) +
         (a < (unsigned char) b) +
         ((long) a < b) +  // expected-warning {{comparison of integers of different signs}}
         ((int) a < b) +  // expected-warning {{comparison of integers of different signs}}
         ((short) a < b) +  // expected-warning {{comparison of integers of different signs}}
         ((signed char) a < b) +  // expected-warning {{comparison of integers of different signs}}
         ((long) a < (unsigned long) b) +  // expected-warning {{comparison of integers of different signs}}
         ((int) a < (unsigned int) b) +  // expected-warning {{comparison of integers of different signs}}
         ((short) a < (unsigned short) b) +
         ((signed char) a < (unsigned char) b) +

         // (A,b)
         (A == (unsigned long) b) +
         (A == (unsigned int) b) +
         (A == (unsigned short) b) +
         (A == (unsigned char) b) +
         ((long) A == b) +
         ((int) A == b) +
         ((short) A == b) +
         ((signed char) A == b) +
         ((long) A == (unsigned long) b) +
         ((int) A == (unsigned int) b) +
         ((short) A == (unsigned short) b) +
         ((signed char) A == (unsigned char) b) +
         (A < (unsigned long) b) +
         (A < (unsigned int) b) +
         (A < (unsigned short) b) +
         (A < (unsigned char) b) +
         ((long) A < b) +
         ((int) A < b) +
         ((short) A < b) +
         ((signed char) A < b) +
         ((long) A < (unsigned long) b) +
         ((int) A < (unsigned int) b) +
         ((short) A < (unsigned short) b) +
         ((signed char) A < (unsigned char) b) +

         // (a,B)
         (a == (unsigned long) B) +
         (a == (unsigned int) B) +
         (a == (unsigned short) B) +
         (a == (unsigned char) B) +
         ((long) a == B) +
         ((int) a == B) +
         ((short) a == B) +
         ((signed char) a == B) +
         ((long) a == (unsigned long) B) +
         ((int) a == (unsigned int) B) +
         ((short) a == (unsigned short) B) +
         ((signed char) a == (unsigned char) B) +
         (a < (unsigned long) B) +  // expected-warning {{comparison of unsigned expression < 0 is always false}}
         (a < (unsigned int) B) +
         (a < (unsigned short) B) +
         (a < (unsigned char) B) +
         ((long) a < B) +
         ((int) a < B) +
         ((short) a < B) +
         ((signed char) a < B) +
         ((long) a < (unsigned long) B) +  // expected-warning {{comparison of unsigned expression < 0 is always false}}
         ((int) a < (unsigned int) B) +  // expected-warning {{comparison of unsigned expression < 0 is always false}}
         ((short) a < (unsigned short) B) +
         ((signed char) a < (unsigned char) B) +

         // (C,b)
         (C == (unsigned long) b) +
         (C == (unsigned int) b) +
         (C == (unsigned short) b) + // expected-warning {{comparison of constant 'C' (65536) with expression of type 'unsigned short' is always false}}
         (C == (unsigned char) b) +  // expected-warning {{comparison of constant 'C' (65536) with expression of type 'unsigned char' is always false}}
         ((long) C == b) +
         ((int) C == b) +
         ((short) C == b) +
         ((signed char) C == b) +
         ((long) C == (unsigned long) b) +
         ((int) C == (unsigned int) b) +
         ((short) C == (unsigned short) b) +
         ((signed char) C == (unsigned char) b) +
         (C < (unsigned long) b) +
         (C < (unsigned int) b) +
         (C < (unsigned short) b) + // expected-warning {{comparison of constant 'C' (65536) with expression of type 'unsigned short' is always false}}
         (C < (unsigned char) b) + // expected-warning {{comparison of constant 'C' (65536) with expression of type 'unsigned char' is always false}}
         ((long) C < b) +
         ((int) C < b) +
         ((short) C < b) +
         ((signed char) C < b) +
         ((long) C < (unsigned long) b) +
         ((int) C < (unsigned int) b) +
         ((short) C < (unsigned short) b) +
         ((signed char) C < (unsigned char) b) +

         // (a,C)
         (a == (unsigned long) C) +
         (a == (unsigned int) C) +
         (a == (unsigned short) C) +
         (a == (unsigned char) C) +
         ((long) a == C) +
         ((int) a == C) +
         ((short) a == C) + // expected-warning {{comparison of constant 'C' (65536) with expression of type 'short' is always false}}
         ((signed char) a == C) + // expected-warning {{comparison of constant 'C' (65536) with expression of type 'signed char' is always false}}
         ((long) a == (unsigned long) C) +
         ((int) a == (unsigned int) C) +
         ((short) a == (unsigned short) C) +
         ((signed char) a == (unsigned char) C) +
         (a < (unsigned long) C) +  // expected-warning {{comparison of integers of different signs}}
         (a < (unsigned int) C) +
         (a < (unsigned short) C) +
         (a < (unsigned char) C) +
         ((long) a < C) +
         ((int) a < C) +
         ((short) a < C) + // expected-warning {{comparison of constant 'C' (65536) with expression of type 'short' is always true}}
         ((signed char) a < C) + // expected-warning {{comparison of constant 'C' (65536) with expression of type 'signed char' is always true}}
         ((long) a < (unsigned long) C) +  // expected-warning {{comparison of integers of different signs}}
         ((int) a < (unsigned int) C) +  // expected-warning {{comparison of integers of different signs}}
         ((short) a < (unsigned short) C) +
         ((signed char) a < (unsigned char) C) +

         // (0x80000,b)
         (0x80000 == (unsigned long) b) +
         (0x80000 == (unsigned int) b) +
         (0x80000 == (unsigned short) b) + // expected-warning {{comparison of constant 524288 with expression of type 'unsigned short' is always false}}
         (0x80000 == (unsigned char) b) + // expected-warning {{comparison of constant 524288 with expression of type 'unsigned char' is always false}}
         ((long) 0x80000 == b) +
         ((int) 0x80000 == b) +
         ((short) 0x80000 == b) +
         ((signed char) 0x80000 == b) +
         ((long) 0x80000 == (unsigned long) b) +
         ((int) 0x80000 == (unsigned int) b) +
         ((short) 0x80000 == (unsigned short) b) +
         ((signed char) 0x80000 == (unsigned char) b) +
         (0x80000 < (unsigned long) b) +
         (0x80000 < (unsigned int) b) +
         (0x80000 < (unsigned short) b) + // expected-warning {{comparison of constant 524288 with expression of type 'unsigned short' is always false}}
         (0x80000 < (unsigned char) b) + // expected-warning {{comparison of constant 524288 with expression of type 'unsigned char' is always false}}
         ((long) 0x80000 < b) +
         ((int) 0x80000 < b) +
         ((short) 0x80000 < b) +
         ((signed char) 0x80000 < b) +
         ((long) 0x80000 < (unsigned long) b) +
         ((int) 0x80000 < (unsigned int) b) +
         ((short) 0x80000 < (unsigned short) b) +
         ((signed char) 0x80000 < (unsigned char) b) +

         // (a,0x80000)
         (a == (unsigned long) 0x80000) +
         (a == (unsigned int) 0x80000) +
         (a == (unsigned short) 0x80000) +
         (a == (unsigned char) 0x80000) +
         ((long) a == 0x80000) +
         ((int) a == 0x80000) +
         ((short) a == 0x80000) + // expected-warning {{comparison of constant 524288 with expression of type 'short' is always false}}
         ((signed char) a == 0x80000) + // expected-warning {{comparison of constant 524288 with expression of type 'signed char' is always false}}
         ((long) a == (unsigned long) 0x80000) +
         ((int) a == (unsigned int) 0x80000) +
         ((short) a == (unsigned short) 0x80000) +
         ((signed char) a == (unsigned char) 0x80000) +
         (a < (unsigned long) 0x80000) +  // expected-warning {{comparison of integers of different signs}}
         (a < (unsigned int) 0x80000) +
         (a < (unsigned short) 0x80000) +
         (a < (unsigned char) 0x80000) +
         ((long) a < 0x80000) +
         ((int) a < 0x80000) +
         ((short) a < 0x80000) + // expected-warning {{comparison of constant 524288 with expression of type 'short' is always true}}
         ((signed char) a < 0x80000) + // expected-warning {{comparison of constant 524288 with expression of type 'signed char' is always true}}
         ((long) a < (unsigned long) 0x80000) +  // expected-warning {{comparison of integers of different signs}}
         ((int) a < (unsigned int) 0x80000) +  // expected-warning {{comparison of integers of different signs}}
         ((short) a < (unsigned short) 0x80000) +
         ((signed char) a < (unsigned char) 0x80000) +

         10
    ;
}

int test1(int i) {
  enum en { zero };
  return i > zero;
}

enum E { e };
void test2(int i, void *vp) {
  if (&i == vp) { } // ok
  if (test1 == vp) { } // expected-warning{{equality comparison between function pointer and void pointer}}
  if (test1 == e) { } // expected-error{{comparison between pointer and integer}}
  if (vp < 0) { } // expected-error {{comparison between pointer and zero}}
  if (test1 < e) { } // expected-error{{comparison between pointer and integer}}
}

// PR7536
static const unsigned int kMax = 0;
int pr7536() {
  return (kMax > 0);
}

// -Wsign-compare should not warn when ?: operands have different signedness.
// This will be caught by -Wsign-conversion
void test3() {
  unsigned long a;
  signed long b;
  (void) (true ? a : b);
  (void) (true ? (unsigned int)a : (signed int)b);
  (void) (true ? b : a);
  (void) (true ? (unsigned char)b : (signed char)a);
}

// Test comparison of short to unsigned.  If tautological compare does not
// trigger, then the signed comparison warning will.
void test4(short s) {
  // A is max short plus 1.  All zero and positive shorts are smaller than it.
  // All negative shorts are cast towards the max unsigned range.  Relation
  // comparisons are possible, but equality comparisons are tautological.
  const unsigned A = 32768;
  void (s < A); // expected-warning{{comparison of integers of different signs: 'short' and 'const unsigned int'}}
  void (s > A); // expected-warning{{comparison of integers of different signs: 'short' and 'const unsigned int'}}
  void (s <= A); // expected-warning{{comparison of integers of different signs: 'short' and 'const unsigned int'}}
  void (s >= A); // expected-warning{{comparison of integers of different signs: 'short' and 'const unsigned int'}}

  void (s == A); // expected-warning{{comparison of constant 32768 with expression of type 'short' is always false}}
  void (s != A); // expected-warning{{comparison of constant 32768 with expression of type 'short' is always true}}

  // When negative one is converted to an unsigned value, it becomes the max
  // unsigned.  Likewise, a negative one short can also be converted to max
  // unsigned.
  const unsigned B = -1;
  void (s < B); // expected-warning{{comparison of integers of different signs: 'short' and 'const unsigned int'}}
  void (s > B); // expected-warning{{comparison of integers of different signs: 'short' and 'const unsigned int'}}
  void (s <= B); // expected-warning{{comparison of integers of different signs: 'short' and 'const unsigned int'}}
  void (s >= B); // expected-warning{{comparison of integers of different signs: 'short' and 'const unsigned int'}}
  void (s == B); // expected-warning{{comparison of integers of different signs: 'short' and 'const unsigned int'}}
  void (s != B); // expected-warning{{comparison of integers of different signs: 'short' and 'const unsigned int'}}

}

void test5(bool b) {
  (void) (b < -1); // expected-warning{{comparison of constant -1 with expression of type 'bool' is always false}}
  (void) (b > -1); // expected-warning{{comparison of constant -1 with expression of type 'bool' is always true}}
  (void) (b == -1); // expected-warning{{comparison of constant -1 with expression of type 'bool' is always false}}
  (void) (b != -1); // expected-warning{{comparison of constant -1 with expression of type 'bool' is always true}}
  (void) (b <= -1); // expected-warning{{comparison of constant -1 with expression of type 'bool' is always false}}
  (void) (b >= -1); // expected-warning{{comparison of constant -1 with expression of type 'bool' is always true}}

  (void) (b < -10); // expected-warning{{comparison of constant -10 with expression of type 'bool' is always false}}
  (void) (b > -10); // expected-warning{{comparison of constant -10 with expression of type 'bool' is always true}}
  (void) (b == -10); // expected-warning{{comparison of constant -10 with expression of type 'bool' is always false}}
  (void) (b != -10); // expected-warning{{comparison of constant -10 with expression of type 'bool' is always true}}
  (void) (b <= -10); // expected-warning{{comparison of constant -10 with expression of type 'bool' is always false}}
  (void) (b >= -10); // expected-warning{{comparison of constant -10 with expression of type 'bool' is always true}}

  (void) (b < 2); // expected-warning{{comparison of constant 2 with expression of type 'bool' is always true}}
  (void) (b > 2); // expected-warning{{comparison of constant 2 with expression of type 'bool' is always false}}
  (void) (b == 2); // expected-warning{{comparison of constant 2 with expression of type 'bool' is always false}}
  (void) (b != 2); // expected-warning{{comparison of constant 2 with expression of type 'bool' is always true}}
  (void) (b <= 2); // expected-warning{{comparison of constant 2 with expression of type 'bool' is always true}}
  (void) (b >= 2); // expected-warning{{comparison of constant 2 with expression of type 'bool' is always false}}

  (void) (b < 10); // expected-warning{{comparison of constant 10 with expression of type 'bool' is always true}}
  (void) (b > 10); // expected-warning{{comparison of constant 10 with expression of type 'bool' is always false}}
  (void) (b == 10); // expected-warning{{comparison of constant 10 with expression of type 'bool' is always false}}
  (void) (b != 10); // expected-warning{{comparison of constant 10 with expression of type 'bool' is always true}}
  (void) (b <= 10); // expected-warning{{comparison of constant 10 with expression of type 'bool' is always true}}
  (void) (b >= 10); // expected-warning{{comparison of constant 10 with expression of type 'bool' is always false}}
}

void test6(signed char sc) {
  (void)(sc < 200); // expected-warning{{comparison of constant 200 with expression of type 'signed char' is always true}}
  (void)(sc > 200); // expected-warning{{comparison of constant 200 with expression of type 'signed char' is always false}}
  (void)(sc <= 200); // expected-warning{{comparison of constant 200 with expression of type 'signed char' is always true}}
  (void)(sc >= 200); // expected-warning{{comparison of constant 200 with expression of type 'signed char' is always false}}
  (void)(sc == 200); // expected-warning{{comparison of constant 200 with expression of type 'signed char' is always false}}
  (void)(sc != 200); // expected-warning{{comparison of constant 200 with expression of type 'signed char' is always true}}

  (void)(200 < sc); // expected-warning{{comparison of constant 200 with expression of type 'signed char' is always false}}
  (void)(200 > sc); // expected-warning{{comparison of constant 200 with expression of type 'signed char' is always true}}
  (void)(200 <= sc); // expected-warning{{comparison of constant 200 with expression of type 'signed char' is always false}}
  (void)(200 >= sc); // expected-warning{{comparison of constant 200 with expression of type 'signed char' is always true}}
  (void)(200 == sc); // expected-warning{{comparison of constant 200 with expression of type 'signed char' is always false}}
  (void)(200 != sc); // expected-warning{{comparison of constant 200 with expression of type 'signed char' is always true}}
}

// Test many signedness combinations.
void test7(unsigned long other) {
  // Common unsigned, other unsigned, constant unsigned
  (void)((unsigned)other != (unsigned long)(0x1ffffffff)); // expected-warning{{true}}
  (void)((unsigned)other != (unsigned long)(0xffffffff));
  (void)((unsigned long)other != (unsigned)(0x1ffffffff));
  (void)((unsigned long)other != (unsigned)(0xffffffff));

  // Common unsigned, other signed, constant unsigned
  (void)((int)other != (unsigned long)(0xffffffffffffffff)); // expected-warning{{different signs}}
  (void)((int)other != (unsigned long)(0x00000000ffffffff)); // expected-warning{{true}}
  (void)((int)other != (unsigned long)(0x000000000fffffff));
  (void)((int)other < (unsigned long)(0x00000000ffffffff));  // expected-warning{{different signs}}
  (void)((int)other == (unsigned)(0x800000000));

  // Common unsigned, other unsigned, constant signed
  (void)((unsigned long)other != (int)(0xffffffff));  // expected-warning{{different signs}}

  // Common unsigned, other signed, constant signed
  // Should not be possible as the common type should also be signed.

  // Common signed, other signed, constant signed
  (void)((int)other != (long)(0xffffffff));  // expected-warning{{true}}
  (void)((int)other != (long)(0xffffffff00000000));  // expected-warning{{true}}
  (void)((int)other != (long)(0xfffffff));
  (void)((int)other != (long)(0xfffffffff0000000));

  // Common signed, other signed, constant unsigned
  (void)((int)other != (unsigned char)(0xffff));
  (void)((int)other != (unsigned char)(0xff));

  // Common signed, other unsigned, constant signed
  (void)((unsigned char)other != (int)(0xff));
  (void)((unsigned char)other != (int)(0xffff));  // expected-warning{{true}}

  // Common signed, other unsigned, constant unsigned
  (void)((unsigned char)other != (unsigned short)(0xff));
  (void)((unsigned char)other != (unsigned short)(0x100)); // expected-warning{{true}}
  (void)((unsigned short)other != (unsigned char)(0xff));
}

void test8(int x) {
  enum E {
    Negative = -1,
    Positive = 1
  };

  (void)((E)x == 1);
  (void)((E)x == -1);
}

void test9(int x) {
  enum E : int {
    Positive = 1
  };
  (void)((E)x == 1);
}

namespace templates {
  template<class T> T max();

  template<> constexpr int max<int>() { return 2147483647; };

  template<typename T>
  bool less_than_max(short num, T value) {
    const T vmax = max<T>();
    return (vmax >= num);  // no warning
  }

  template<typename T>
  bool less_than_max(short num) {
    // This should trigger one warning on the template pattern, and not a
    // warning per specialization.
    return num < max<int>();  // expected-warning{{comparison of constant 2147483647 with expression of type 'short' is always true}}
  }

  void test10(short num, int x) {
    less_than_max(num, x);
    less_than_max<int>(num);
    less_than_max<long>(num);
    less_than_max<short>(num);
  }

  template<typename T>
  inline bool less_than_zero(T num, T value) {
    return num < 0;  // no warning
  }

  template<typename T>
  inline bool less_than_zero(unsigned num) {
    // This should trigger one warning on the template pattern, and not a
    // warning per specialization.
    return num < 0;  // expected-warning{{comparison of unsigned expression < 0 is always false}}
  }

  void test11(unsigned num) {
    less_than_zero(num, num);
    less_than_zero<int>(num);
    less_than_zero<long>(num);
    less_than_zero<short>(num);
  }

  template<unsigned n> bool compare(unsigned k) { return k >= n; }

  void test12() {
    compare<0>(42);
  }

  struct A { static int x; };
  struct B { static int x; };
  typedef A otherA;

  template <typename T>
  void testx() {
    if (A::x == T::x &&  // no warning
        A::x == otherA::x)  // expected-warning{{self-comparison always evaluates to true}}
      return;
  }

  void test13() {
    testx<A>();
    testx<B>();
  }
}
