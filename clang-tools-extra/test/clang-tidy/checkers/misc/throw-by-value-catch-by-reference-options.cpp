// RUN: %check_clang_tidy %s misc-throw-by-value-catch-by-reference %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     misc-throw-by-value-catch-by-reference.WarnOnLargeObject: true, \
// RUN:     misc-throw-by-value-catch-by-reference.MaxSize: 200, \
// RUN:     misc-throw-by-value-catch-by-reference.CheckThrowTemporaries: false \
// RUN:   }}" -- -fcxx-exceptions

struct LargeTrivial {
  char data[100];
};

struct SmallTrivial {
  char data[10];
};

struct NonTrivial {
  NonTrivial() {}
  NonTrivial(const NonTrivial &) {}
  char data[100];
};

void testLargeTrivial() {
  try {
    throw LargeTrivial();
  } catch (LargeTrivial e) {
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: catch handler catches by value; should catch by reference instead [misc-throw-by-value-catch-by-reference]
  }
}

void testSmallTrivial() {
  try {
    throw SmallTrivial();
  } catch (SmallTrivial e) {
    // Should not warn (80 < 200)
  }
}

void testNonTrivial() {
  try {
    throw NonTrivial();
  } catch (NonTrivial e) {
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: catch handler catches by value; should catch by reference instead [misc-throw-by-value-catch-by-reference]
  }
}

void testCheckThrowTemporaries() {
  LargeTrivial lvalue;
  throw lvalue;
  // Should not warn when CheckThrowTemporaries is false
}
