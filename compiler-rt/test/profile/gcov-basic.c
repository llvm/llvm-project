// RUN: mkdir -p %t.dir && cd %t.dir

/// gcov 3.4 redesigned the format and changed the extension from .da to .gcda
// RUN: %clang --coverage -Xclang -coverage-version='304*' %s -o %t
// RUN: rm -f gcov-basic.gcda && %run %t && %run %t a
// RUN: llvm-cov gcov -t gcov-basic.gcno | FileCheck %s

/// r173147: split checksum into cfg checksum and line checksum.
// RUN: %clang --coverage -Xclang -coverage-version='407*' %s -o %t
// RUN: rm -f gcov-basic.gcda && %run %t && %run %t a
// RUN: llvm-cov gcov -t gcov-basic.gcno | FileCheck %s

/// r189778: the exit block moved from the last to the second.
// RUN: %clang --coverage -Xclang -coverage-version='408*' %s -o %t
// RUN: rm -f gcov-basic.gcda && %run %t && %run %t a
// RUN: llvm-cov gcov -t gcov-basic.gcno

/// PR gcov-profile/48463
// RUN: %clang --coverage -Xclang -coverage-version='800*' %s -o %t
// RUN: rm -f gcov-basic.gcda && %run %t && %run %t a
// RUN: llvm-cov gcov -t gcov-basic.gcno

/// PR gcov-profile/84846, r269678
// RUN: %clang --coverage -Xclang -coverage-version='900*' %s -o %t
// RUN: rm -f gcov-basic.gcda && %run %t && %run %t a
// RUN: llvm-cov gcov -t gcov-basic.gcno

// CHECK: Runs:2

int main(int argc, char *argv[]) { // CHECK:      2: [[@LINE]]:int main
  if (argc > 1)                    // CHECK-NEXT: 2: [[@LINE]]:
    puts("hello");                 // CHECK-NEXT: 1: [[@LINE]]:
  return 0;                        // CHECK-NEXT: 2: [[@LINE]]:
}
