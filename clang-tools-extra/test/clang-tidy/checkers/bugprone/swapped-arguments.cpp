// RUN: %check_clang_tidy %s bugprone-swapped-arguments %t

void F(int, double);

int SomeFunction();

template <typename T, typename U>
void G(T a, U b) {
  F(a, b); // no-warning
  F(2.0, 4);
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: argument with implicit conversion from 'double' to 'int' followed by argument converted from 'int' to 'double', potentially swapped arguments.
// CHECK-FIXES: F(4, 2.0)
}

void funShortFloat(short, float);
void funFloatFloat(float, float);
void funBoolShort(bool, short);
void funBoolFloat(bool, float);

void foo() {
  F(1.0, 3);
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: argument with implicit conversion from 'double' to 'int' followed by argument converted from 'int' to 'double', potentially swapped arguments.
// CHECK-FIXES: F(3, 1.0)

#define M(x, y) x##y()

  double b = 1.0;
  F(b, M(Some, Function));
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: argument with implicit conversion from 'double' to 'int' followed by argument converted from 'int' to 'double', potentially swapped arguments.
// CHECK-FIXES: F(M(Some, Function), b);

#define N F(b, SomeFunction())

  N;
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: argument with implicit conversion from 'double' to 'int' followed by argument converted from 'int' to 'double', potentially swapped arguments.
// In macro, don't emit fixits.
// CHECK-FIXES: #define N F(b, SomeFunction())

  G(b, 3);
  G(3, 1.0);
  G(0, 0);

  F(1.0, 1.0);    // no-warning
  F(3, 1.0);      // no-warning
  F(true, false); // no-warning
  F(0, 'c');      // no-warning

#define APPLY(f, x, y) f(x, y)
  APPLY(F, 1.0, 3);
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: argument with implicit conversion from 'double' to 'int' followed by argument converted from 'int' to 'double', potentially swapped arguments.
// CHECK-FIXES: APPLY(F, 3, 1.0);

#define PARAMS 1.0, 3
#define CALL(P) F(P)
  CALL(PARAMS);
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: argument with implicit conversion from 'double' to 'int' followed by argument converted from 'int' to 'double', potentially swapped arguments.
// In macro, don't emit fixits.

  funShortFloat(5.0, 1U);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: argument with implicit conversion from 'double' to 'short' followed by argument converted from 'unsigned int' to 'float', potentially swapped arguments.
  // CHECK-FIXES: funShortFloat(1U, 5.0);
  funShortFloat(5.0, 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: argument with implicit conversion from 'double' to 'short' followed by argument converted from 'int' to 'float', potentially swapped arguments.
  // CHECK-FIXES: funShortFloat(1, 5.0);
  funShortFloat(5.0f, 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: argument with implicit conversion from 'float' to 'short' followed by argument converted from 'int' to 'float', potentially swapped arguments.
  // CHECK-FIXES: funShortFloat(1, 5.0f);

  funFloatFloat(1.0f, 2.0);

  funBoolShort(1U, true);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: argument with implicit conversion from 'unsigned int' to 'bool' followed by argument converted from 'bool' to 'short', potentially swapped arguments.
  // CHECK-FIXES: funBoolShort(true, 1U);

  funBoolFloat(1.0, true);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: argument with implicit conversion from 'double' to 'bool' followed by argument converted from 'bool' to 'float', potentially swapped arguments.
  // CHECK-FIXES: funBoolFloat(true, 1.0);
}
