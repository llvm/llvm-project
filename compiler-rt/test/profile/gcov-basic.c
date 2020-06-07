// RUN: mkdir -p %t.dir && cd %t.dir

/// gcov 3.4 redesigned the format and changed the extension from .da to .gcda
// RUN: %clang --coverage -Xclang -coverage-version='304*' %s -o %t
// RUN: rm -f gcov-basic.gcda && %run %t
// RUN: llvm-cov gcov -t gcov-basic.gcno | FileCheck %s

/// r173147: split checksum into cfg checksum and line checksum.
// RUN: %clang --coverage -Xclang -coverage-version='407*' %s -o %t
// RUN: rm -f gcov-basic.gcda && %run %t
// RUN: llvm-cov gcov -t gcov-basic.gcno | FileCheck %s

/// r189778: the exit block moved from the last to the second.
// RUN: %clang --coverage -Xclang -coverage-version='408*' %s -o %t
// RUN: rm -f gcov-basic.gcda && %run %t
// RUN: llvm-cov gcov -t gcov-basic.gcno

int main() {       // CHECK:      1: [[@LINE]]:int main
  return 0;        // CHECK-NEXT: 1: [[@LINE]]:
}
