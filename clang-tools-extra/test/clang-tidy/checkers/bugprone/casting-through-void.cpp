// RUN: %check_clang_tidy %s bugprone-casting-through-void %t

using V = void*;
using CV = const void*;

int i = 100;
double d = 100;
const int ci = 100;
const double cd = 100;

void normal_test() {
  static_cast<int *>(static_cast<void *>(&d));
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: do not cast 'double *' to 'int *' through 'void *' [bugprone-casting-through-void]
  static_cast<int *>(static_cast<V>(&d));
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: do not cast 'double *' to 'int *' through 'V' (aka 'void *') [bugprone-casting-through-void]
  static_cast<int *>(static_cast<void *>(&i));
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: do not cast 'int *' to 'int *' through 'void *' [bugprone-casting-through-void]

  static_cast<void *>(static_cast<void *>(&i));
}

void const_pointer_test() {
  static_cast<int *const>(static_cast<void *>(&d));
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: do not cast 'double *' to 'int *const' through 'void *' [bugprone-casting-through-void]
  static_cast<int *const>(static_cast<V>(&d));
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: do not cast 'double *' to 'int *const' through 'V' (aka 'void *') [bugprone-casting-through-void]
  static_cast<int *const>(static_cast<void *>(&i));
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: do not cast 'int *' to 'int *const' through 'void *' [bugprone-casting-through-void]

  static_cast<void *const>(static_cast<void *>(&i));
}

void const_test() {
  static_cast<const int *>(static_cast<const void *>(&d));
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: do not cast 'double *' to 'const int *' through 'const void *' [bugprone-casting-through-void]
  static_cast<const int *>(static_cast<const V>(&d));
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: do not cast 'double *' to 'const int *' through 'const V' (aka 'void *const') [bugprone-casting-through-void]
  static_cast<const int *>(static_cast<const void *>(&i));
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: do not cast 'int *' to 'const int *' through 'const void *' [bugprone-casting-through-void]

  static_cast<const void *>(static_cast<const void *>(&i));

  static_cast<const int *>(static_cast<const void *>(&cd));
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: do not cast 'const double *' to 'const int *' through 'const void *' [bugprone-casting-through-void]
  static_cast<const int *>(static_cast<const CV>(&cd));
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: do not cast 'const double *' to 'const int *' through 'const CV' (aka 'const void *const') [bugprone-casting-through-void]
  static_cast<const int *>(static_cast<const void *>(&ci));
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: do not cast 'const int *' to 'const int *' through 'const void *' [bugprone-casting-through-void]

  static_cast<const void *>(static_cast<const void *>(&ci));
}


void reinterpret_cast_test() {
  static_cast<int *>(reinterpret_cast<void *>(&d));
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: do not cast 'double *' to 'int *' through 'void *' [bugprone-casting-through-void]
  reinterpret_cast<int *>(static_cast<void *>(&d));
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: do not cast 'double *' to 'int *' through 'void *' [bugprone-casting-through-void]
  reinterpret_cast<int *>(reinterpret_cast<void *>(&d));
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: do not cast 'double *' to 'int *' through 'void *' [bugprone-casting-through-void]

  static_cast<void *>(reinterpret_cast<void *>(&i));
  reinterpret_cast<void *>(reinterpret_cast<void *>(&i));
  reinterpret_cast<void *>(static_cast<void *>(&i));
}

void c_style_cast_test() {
  static_cast<int *>((void *)&d);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: do not cast 'double *' to 'int *' through 'void *' [bugprone-casting-through-void]
  (int *)(void *)&d;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: do not cast 'double *' to 'int *' through 'void *' [bugprone-casting-through-void]
  static_cast<int *>((void *)&d);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: do not cast 'double *' to 'int *' through 'void *' [bugprone-casting-through-void]

  static_cast<void *>((void *)&i);
}

struct A {
   A(void*);
};
using I = int *;
void cxx_functional_cast() {
  A(static_cast<void*>(&d));
  I(static_cast<void*>(&d));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: do not cast 'double *' to 'I' (aka 'int *') through 'void *' [bugprone-casting-through-void]
}

void bit_cast() {
  __builtin_bit_cast(int *, static_cast<void *>(&d));
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: do not cast 'double *' to 'int *' through 'void *' [bugprone-casting-through-void]
}
