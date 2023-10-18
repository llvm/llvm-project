// RUN: %check_clang_tidy %s bugprone-casting-through-void %t

using T = void*;
using CT = const void*;

int i = 100;
double d = 100;
const int ci = 100;
const double cd = 100;

void normal_test() {
  static_cast<int *>(static_cast<void *>(&d));
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: do not cast 'double *' to 'int *' through 'void*' [bugprone-casting-through-void]

  static_cast<int *>(static_cast<T>(&d));
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: do not cast 'double *' to 'int *' through 'void*' [bugprone-casting-through-void]

  static_cast<int *>(static_cast<void *>(&i));
  static_cast<void *>(static_cast<void *>(&i));
}

void const_pointer_test() {
  static_cast<int *const>(static_cast<void *>(&d));
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: do not cast 'double *' to 'int *const' through 'void*' [bugprone-casting-through-void]

  static_cast<int *const>(static_cast<T>(&d));
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: do not cast 'double *' to 'int *const' through 'void*' [bugprone-casting-through-void]

  static_cast<int *const>(static_cast<void *>(&i));
  static_cast<void *const>(static_cast<void *>(&i));
}

void const_test() {
  static_cast<const int *>(static_cast<const void *>(&d));
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: do not cast 'double *' to 'const int *' through 'void*' [bugprone-casting-through-void]

  static_cast<const int *>(static_cast<const T>(&d));
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: do not cast 'double *' to 'const int *' through 'void*' [bugprone-casting-through-void]

  static_cast<const int *>(static_cast<const void *>(&i));
  static_cast<const void *>(static_cast<const void *>(&i));

  static_cast<const int *>(static_cast<const void *>(&cd));
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: do not cast 'const double *' to 'const int *' through 'void*' [bugprone-casting-through-void]

  static_cast<const int *>(static_cast<const CT>(&cd));
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: do not cast 'const double *' to 'const int *' through 'void*' [bugprone-casting-through-void]

  static_cast<const int *>(static_cast<const void *>(&ci));
  static_cast<const void *>(static_cast<const void *>(&ci));
}


void reinterpret_cast_test() {
  static_cast<int *>(reinterpret_cast<void *>(&d));
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: do not cast 'double *' to 'int *' through 'void*' [bugprone-casting-through-void]
  reinterpret_cast<int *>(static_cast<void *>(&d));
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: do not cast 'double *' to 'int *' through 'void*' [bugprone-casting-through-void]
  reinterpret_cast<int *>(reinterpret_cast<void *>(&d));
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: do not cast 'double *' to 'int *' through 'void*' [bugprone-casting-through-void]

  static_cast<int *>(reinterpret_cast<void *>(&i));
  static_cast<void *>(reinterpret_cast<void *>(&i));
  reinterpret_cast<int *>(reinterpret_cast<void *>(&i));
  reinterpret_cast<void *>(reinterpret_cast<void *>(&i));
  static_cast<int *>(reinterpret_cast<void *>(&i));
  static_cast<void *>(reinterpret_cast<void *>(&i));
}

void c_style_cast_test() {
  static_cast<int *>((void *)&d);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: do not cast 'double *' to 'int *' through 'void*' [bugprone-casting-through-void]
  (int *)(void *)&d;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: do not cast 'double *' to 'int *' through 'void*' [bugprone-casting-through-void]
  static_cast<int *>((void *)&d);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: do not cast 'double *' to 'int *' through 'void*' [bugprone-casting-through-void]

  static_cast<int *>(reinterpret_cast<void *>(&i));
  static_cast<void *>(reinterpret_cast<void *>(&i));
}

struct A {
   A(void*);
};
void cxx_functional_cast() {
  A(static_cast<void*>(&i));
}

void bit_cast() {
  __builtin_bit_cast(int *, static_cast<void *>(&i));
}
