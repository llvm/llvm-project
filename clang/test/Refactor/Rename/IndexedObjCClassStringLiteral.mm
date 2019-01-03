@interface Test

@end

void foo() {
  "Test"; // CHECK: string-literal [[@LINE]]:4 -> [[@LINE]]:8
  @"Test"; // CHECK: string-literal [[@LINE]]:5 -> [[@LINE]]:9
  u8"Test";  // CHECK: string-literal [[@LINE]]:6 -> [[@LINE]]:10
  // CHECK-NOT: string-literal
  "Test.h";
  " Test ";
  "test";
  "TEST";
}

// RUN: clang-refactor-test rename-indexed-file -name=Test -new-name=foo -indexed-file=%s -indexed-at=1:12 -indexed-symbol-kind=objc-class %s -std=c++11 | FileCheck %s
// It should be possible to find a string-literal in a file without any indexed occurrences:
// RUN: clang-refactor-test rename-indexed-file -name=Test -new-name=foo -indexed-file=%s -indexed-symbol-kind=objc-class %s -std=c++11 | FileCheck %s

// RUN: clang-refactor-test rename-indexed-file -name=Test -new-name=foo -indexed-file=%s -indexed-at=1:12 %s -std=c++11 | FileCheck --check-prefix=NOTCLASS %s
// NOTCLASS-NOT: string-literal
