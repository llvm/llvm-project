// RUN: %check_clang_tidy %s bugprone-branch-clone %t --

int test_asm1(int argc, char**) {
  if (argc > 1) {
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: if with identical then and else branches [bugprone-branch-clone]
    __asm__ volatile(
      "addi %0, %0, -1"
      : "+r" (argc)
      );
  } else {
// CHECK-MESSAGES: :[[@LINE-1]]:5: note: else branch starts here
    __asm__ volatile(
      "addi %0, %0, -1"
      : "+r" (argc)
      );
  }
  return argc;
}

int test_asm2(int argc, char**) {
  if (argc > 1) { // no-warning
    __asm__ volatile(
      "addi %0, %0, -1"
      : "+r" (argc)
      );
  } else {
    __asm__ volatile(
      "addi %0, %0, -2"
      : "+r" (argc)
      );
  }
  return argc;
}

int test_asm3(int argc, char**) {
  int Test1 = 0;
  if (argc > 1) {
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: if with identical then and else branches [bugprone-branch-clone]
    __asm__ volatile(
      "add %w0, %w0, -1"
      : "+r" (argc)
      : "r" (Test1)
      : "w0"
      );
  } else {
// CHECK-MESSAGES: :[[@LINE-1]]:5: note: else branch starts here
    __asm__ volatile(
      "add %w0, %w0, -1"
      : "+r" (argc)
      : "r" (Test1)
      : "w0"
      );
  }
  return argc;
}

int test_asm4(int argc, char**) {
  int Test1 = 0;
  if (argc > 1) { // no-warning
    __asm__ volatile(
      "add %w0, %w0, -1"
      : "+r" (argc)
      : "r" (Test1)
      : "w0"
      );
  } else {
    __asm__ volatile(
      "add %w0, %w0, -1" 
      : "+r" (argc)
      : "r" (Test1)
      : "w1"
      );
  }
  return argc;
}