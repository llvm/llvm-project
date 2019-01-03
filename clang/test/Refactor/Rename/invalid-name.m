// XFAIL: *
// TODO: Remove if unused
// Note: the run lines follow their respective tests, since line/column
// matter in this test

int variable = 0; // CHECK1: rename [[@LINE]]:5 -> [[@LINE]]:13

// RUN: clang-refactor-test rename-initiate -at=%s:4:5 -new-name=class %s | FileCheck --check-prefix=CHECK1 %s
// RUN: not clang-refactor-test rename-initiate -at=%s:4:5 -new-name=class %s -x objective-c++ 2>&1 | FileCheck --check-prefix=CHECK-ERR %s
// CHECK-ERR: error: invalid new name

// RUN: not clang-refactor-test rename-initiate -at=%s:4:5 -new-name=int %s 2>&1 | FileCheck --check-prefix=CHECK-ERR %s

@interface I

- (void)some:(int)x selector:(int)y; // CHECK2: rename [[@LINE]]:9 -> [[@LINE]]:13, [[@LINE]]:21 -> [[@LINE]]:29

@end

// RUN: clang-refactor-test rename-initiate -at=%s:14:9 -new-name=struct:void %s | FileCheck --check-prefix=CHECK2 %s

// RUN: not clang-refactor-test rename-initiate -at=%s:14:9 -new-name=hello:123 %s 2>&1 | FileCheck --check-prefix=CHECK-ERR %s
// RUN: not clang-refactor-test rename-initiate -at=%s:14:9 -new-name=+:test %s 2>&1 | FileCheck --check-prefix=CHECK-ERR %s
