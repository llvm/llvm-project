struct S1 { }; // CHECK1: rename [[@LINE]]:8 -> [[@LINE]]:10

typedef struct S2 { } S3; // CHECK2: rename [[@LINE]]:16 -> [[@LINE]]:18
// CHECK3: rename [[@LINE-1]]:23 -> [[@LINE-1]]:25

void func(struct S1, // CHECK1-NEXT: rename [[@LINE]]:18 -> [[@LINE]]:20
          struct S2, // CHECK2-NEXT: rename [[@LINE]]:18 -> [[@LINE]]:20
          S3) { // CHECK3-NEXT: rename [[@LINE]]:11 -> [[@LINE]]:13
}

// RUN: clang-refactor-test rename-initiate -at=%s:1:8 -at=%s:6:18 -new-name=Bar %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:3:16 -at=%s:7:18 -new-name=Bar %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test rename-initiate -at=%s:3:23 -at=%s:8:11 -new-name=Bar %s | FileCheck --check-prefix=CHECK3 %s

// RUN: clang-refactor-test rename-initiate -at=%s:1:8 -at=%s:6:18 -new-name=Bar %s -x c | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:3:16 -at=%s:7:18 -new-name=Bar %s -x c | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test rename-initiate -at=%s:3:23 -at=%s:8:11 -new-name=Bar %s -x c | FileCheck --check-prefix=CHECK3 %s

// RUN: not clang-refactor-test rename-initiate -at=%s:3:9 -at=%s:6:11 -at=%s:7:11 -new-name=Bar %s 2>&1 | FileCheck --check-prefix=ERROR %s
// RUN: not clang-refactor-test rename-initiate -at=%s:3:9 -at=%s:6:11 -at=%s:7:11 -new-name=Bar %s -x c 2>&1 | FileCheck --check-prefix=ERROR %s
// ERROR: could not rename symbol at the given location
