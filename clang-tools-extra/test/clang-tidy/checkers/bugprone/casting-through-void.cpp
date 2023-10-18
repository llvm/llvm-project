// RUN: %check_clang_tidy %s bugprone-casting-through-void %t

using T = void*;

void test() {
  int i = 100;
  double d = 100;

  static_cast<int *>(static_cast<void *>(&d));
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: do not cast 'double *' to 'int *' through 'void*' [bugprone-casting-through-void]

  static_cast<int *>(static_cast<T>(&d));
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: do not cast 'double *' to 'int *' through 'void*' [bugprone-casting-through-void]

  static_cast<int *>(static_cast<void *>(&i));

  static_cast<void *>(static_cast<void *>(&i));
}
