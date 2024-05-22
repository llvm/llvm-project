// RUN: %check_clang_tidy %s bugprone-too-small-loop-variable %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:             {bugprone-too-small-loop-variable.MagnitudeBitsUpperLimit: 1024}}" \
// RUN:   -- --target=x86_64-linux

long size() { return 294967296l; }

////////////////////////////////////////////////////////////////////////////////
/// Test cases correctly caught by bugprone-too-small-loop-variable.

void voidBadForLoop() {
  for (int i = 0; i < size(); ++i) {
    // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: loop variable has narrower type 'int' than iteration's upper bound 'long' [bugprone-too-small-loop-variable]
  }
}

void voidBadForLoop2() {
  for (int i = 0; i < size() + 10; ++i) {
    // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: loop variable has narrower type 'int' than iteration's upper bound 'long' [bugprone-too-small-loop-variable]
  }
}

void voidBadForLoop3() {
  for (int i = 0; i <= size() - 1; ++i) {
    // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: loop variable has narrower type 'int' than iteration's upper bound 'long' [bugprone-too-small-loop-variable]
  }
}

void voidBadForLoop4() {
  for (int i = 0; size() > i; ++i) {
    // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: loop variable has narrower type 'int' than iteration's upper bound 'long' [bugprone-too-small-loop-variable]
  }
}

void voidBadForLoop5() {
  for (int i = 0; size() - 1 >= i; ++i) {
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: loop variable has narrower type 'int' than iteration's upper bound 'long' [bugprone-too-small-loop-variable]
  }
}

void voidBadForLoop6() {
  int i = 0;
  for (; i < size(); ++i) {
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: loop variable has narrower type 'int' than iteration's upper bound 'long' [bugprone-too-small-loop-variable]
  }
}

void voidBadForLoop7() {
    struct Int  {
        int value;
    } i;

  for (i.value = 0; i.value < size(); ++i.value) {
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: loop variable has narrower type 'int' than iteration's upper bound 'long' [bugprone-too-small-loop-variable]
  }
}

void voidForLoopUnsignedBound() {
  unsigned size = 3147483647;
  for (int i = 0; i < size; ++i) {
    // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: loop variable has narrower type 'int' than iteration's upper bound 'unsigned int' [bugprone-too-small-loop-variable]
  }
}

// The iteration's upper bound has a template dependent value.
template <long size>
void doSomething() {
  for (short i = 0; i < size; ++i) {
    // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: loop variable has narrower type 'short' than iteration's upper bound 'long' [bugprone-too-small-loop-variable]
  }
}

// The iteration's upper bound has a template dependent type.
template <class T>
void doSomething() {
  for (T i = 0; i < size(); ++i) {
    // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: loop variable has narrower type 'short' than iteration's upper bound 'long' [bugprone-too-small-loop-variable]
  }
}

void voidForLoopInstantiation() {
  // This line does not trigger the warning.
  doSomething<long>();
  // This one triggers the warning.
  doSomething<short>();
}

// A suspicious function used in a macro.
#define SUSPICIOUS_SIZE (size())
void voidBadForLoopWithMacroBound() {
  for (short i = 0; i < SUSPICIOUS_SIZE; ++i) {
    // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: loop variable has narrower type 'short' than iteration's upper bound 'long' [bugprone-too-small-loop-variable]
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Correct loops: we should not warn here.

// A simple use case when both expressions have the same type.
void voidGoodForLoop() {
  for (long i = 0; i < size(); ++i) { // no warning
  }
}

// Other use case where both expressions have the same type,
// but short expressions are converted to int by the compare operator.
void voidGoodForLoop2() {
  short loopCond = 10;
  for (short i = 0; i < loopCond; ++i) { // no warning
  }
}

// Because of the integer literal, the iteration's upper bound is int, but we suppress the warning here.
void voidForLoopShortPlusLiteral() {
  short size = 30000;
  for (short i = 0; i <= (size - 1); ++i) { // no warning
  }
}

// Addition of two short variables results in an int value, but we suppress this to avoid false positives.
void voidForLoopShortPlusShort() {
  short size = 256;
  short increment = 14;
  for (short i = 0; i < size + increment; ++i) { // no warning
  }
}

// In this test case we have different integer types, but here the loop variable has the bigger type.
// The iteration's bound is cast implicitly, not the loop variable.
void voidForLoopBoundImplicitCast() {
  short start = 256;
  short end = 14;
  for (int i = start; i >= end; --i) { // no warning
  }
}

// Range based loop and other iterator based loops are ignored by this check.
void voidRangeBasedForLoop() {
  int array[] = {1, 2, 3, 4, 5};
  for (const int &i : array) { // no warning
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Future possibilites to improve the check.

// False positive: because of the int literal, iteration's upper bound has int type.
void voidForLoopFalsePositive() {
  short size = 30000;
  bool cond = false;
  for (short i = 0; i < (cond ? 0 : size); ++i) {
    // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: loop variable has narrower type 'short' than iteration's upper bound 'int' [bugprone-too-small-loop-variable]
  }
}

void voidForLoopFalsePositive2() {
  short size = 30000;
  bool cond = false;
  for (short i = 0; i < (!cond ? size : 0); ++i) {
    // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: loop variable has narrower type 'short' than iteration's upper bound 'int' [bugprone-too-small-loop-variable]
  }
}

// False positive: The loop bound expression contains nested binary operators.
void voidForLoopFalsePositive3() {
  short number = 30000;
  for (short i = 0; i < ((number & 0x7f) + 1); ++i) {
    // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: loop variable has narrower type 'short' than iteration's upper bound 'int' [bugprone-too-small-loop-variable]
  }
}

// TODO: handle while loop.
void voidBadWhileLoop() {
  short i = 0;
  while (i < size()) { // missing warning
    ++i;
  }
}

// TODO: handle do-while loop.
void voidBadDoWhileLoop() {
  short i = 0;
  do {
    ++i;
  } while (i < size()); // missing warning
}

// TODO: handle complex loop conditions.
void voidComplexForCond() {
  bool additionalCond = true;
  for (int i = 0; i < size() && additionalCond; ++i) { // missing warning
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Suspicious test cases ingored by this check.

// Test case with a reverse iteration.
// This is caught by -Wimplicit-int-conversion.
void voidReverseForLoop() {
  for (short i = size() - 1; i >= 0; --i) { // no warning
  }
}

// Macro defined literals are used inside the loop condition.
#define SIZE 125
#define SIZE2 (SIZE + 1)
void voidForLoopWithMacroBound() {
  for (short i = 0; i < SIZE2; ++i) { // no warning
  }
}

// A suspicious loop is not caught if the iteration's upper bound is a literal.
void voidForLoopWithLiteralBound() {
  for (short i = 0; i < 125; ++i) { // no warning
  }
}

// The used literal leads to an infinite loop.
// This is caught by -Wtautological-constant-out-of-range-compare.
void voidForLoopWithBigLiteralBound() {
  for (short i = 0; i < 294967296l; ++i) { // no warning
  }
}

enum eSizeType {
  START,
  Y,
  END
};

// A suspicious loop is not caught if the iteration's upper bound is an enum value.
void voidForLoopWithEnumBound() {
  for (short i = eSizeType::START; i < eSizeType::END; ++i) { // no warning
  }
}

enum eSizeType2 : long {
  START2 = 294967296l,
  Y2,
  END2
};

// The used enum value leads to an infinite loop.
// This is caught by -Wtautological-constant-out-of-range-compare.
void voidForLoopWithBigEnumBound() {
  for (short i = eSizeType2::START2; i < eSizeType2::END2; ++i) { // no warning
  }
}

// A suspicious loop is not caught if the iteration's upper bound is a constant variable.
void voidForLoopWithConstBound() {
  const long size = 252l;
  for (short i = 0; i < size; ++i) { // no warning
  }
}

// The used constant variable leads to an infinite loop.
// This is caught by -Wtautological-constant-out-of-range-compare.
void voidForLoopWithBigConstBound() {
  const long size = 294967296l;
  for (short i = 0; i < size; ++i) { // no warning
  }
}

// Should detect proper size of upper bound bitfield
void voidForLoopWithBitfieldOnUpperBound() {
  struct StructWithBitField {
      unsigned bitfield : 5;
  } value = {};

  for(unsigned char i = 0U; i < value.bitfield; ++i) { // no warning
  }
}

// Should detect proper size of loop variable bitfield
void voidForLoopWithBitfieldOnLoopVar() {
  struct StructWithBitField {
      unsigned bitfield : 9;
  } value = {};

  unsigned char upperLimit = 100U;

  for(value.bitfield = 0U; value.bitfield < upperLimit; ++value.bitfield) {
  }
}

// Should detect proper size of loop variable and upper bound
void voidForLoopWithBitfieldOnLoopVarAndUpperBound() {
  struct StructWithBitField {
      unsigned var : 5, limit : 4;
  } value = {};

  for(value.var = 0U; value.var < value.limit; ++value.var) {
  }
}

// Should detect proper size of loop variable and upper bound on integers
void voidForLoopWithBitfieldOnLoopVarAndUpperBoundOnInt() {
  struct StructWithBitField {
      unsigned var : 5;
      int limit : 6;
  } value = {};

  for(value.var = 0U; value.var < value.limit; ++value.var) {
  }
}

void badForLoopWithBitfieldOnUpperBound() {
  struct StructWithBitField {
      unsigned bitfield : 9;
  } value = {};

  for(unsigned char i = 0U; i < value.bitfield; ++i) {
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: loop variable has narrower type 'unsigned char' than iteration's upper bound 'unsigned int:9' [bugprone-too-small-loop-variable]
  }
}

void badForLoopWithBitfieldOnLoopVar() {
  struct StructWithBitField {
      unsigned bitfield : 7;
  } value = {};

  unsigned char upperLimit = 100U;

  for(value.bitfield = 0U; value.bitfield < upperLimit; ++value.bitfield) {
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: loop variable has narrower type 'unsigned int:7' than iteration's upper bound 'unsigned char' [bugprone-too-small-loop-variable]
  }
}

void badForLoopWithBitfieldOnLoopVarAndUpperBound() {
  struct StructWithBitField {
      unsigned var : 5, limit : 6;
  } value = {};

  for(value.var = 0U; value.var < value.limit; ++value.var) {
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: loop variable has narrower type 'unsigned int:5' than iteration's upper bound 'unsigned int:6' [bugprone-too-small-loop-variable]
  }
}

void badForLoopWithBitfieldOnLoopVarOnIntAndUpperBound() {
  struct StructWithBitField {
      int var : 5;
      unsigned limit : 5;
  } value = {};

  for(value.var = 0U; value.var < value.limit; ++value.var) {
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: loop variable has narrower type 'int:5' than iteration's upper bound 'unsigned int:5' [bugprone-too-small-loop-variable]
  }
}

void badForLoopWithBitfieldOnLoopVarAndUpperBoundOnInt() {
  struct StructWithBitField {
      unsigned var : 5;
      int limit : 7;
  } value = {};

  for(value.var = 0U; value.var < value.limit; ++value.var) {
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: loop variable has narrower type 'unsigned int:5' than iteration's upper bound 'int:7' [bugprone-too-small-loop-variable]
  }
}

void badForLoopWithBitfieldOnLoopVarAndUpperBoundOnPtr() {
  struct StructWithBitField {
      unsigned var : 5, limit : 6;
  } value = {};

  StructWithBitField* ptr = &value;

  for(ptr->var = 0U; ptr->var < ptr->limit; ++ptr->var) {
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: loop variable has narrower type 'unsigned int:5' than iteration's upper bound 'unsigned int:6' [bugprone-too-small-loop-variable]
  }
}

void goodForLoopWithBitfieldOnUpperBoundOnly() {
  struct S {
    int x : 4;
  } s;

  for (int i = 10; i > s.x; --i) {
  }
}

void goodForLoopWithIntegersOnUpperBoundOnly() {
  struct S {
    short x;
  } s;

  for (int i = 10; i > s.x; --i) {
  }
}
