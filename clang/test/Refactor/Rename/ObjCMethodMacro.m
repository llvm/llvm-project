#define FOO foo // CHECK1-NOT: rename [[@LINE]]
// CHECK1-NOT: macro [[@LINE-1]]

@interface I

- (void)FOO; // CHECK1: macro [[@LINE]]:9 -> [[@LINE]]:9
- (void)foo: (int)x FOO: (int)y; // CHECK2: macro [[@LINE]]:21 -> [[@LINE]]:21

@end

@implementation I

- (void)foo { // CHECK1-NEXT: rename [[@LINE]]:9 -> [[@LINE]]:12
  [self FOO: 1 FOO: 2]; // CHECK2-NEXT: macro [[@LINE]]:9 -> [[@LINE]]:9
}

- (void)foo: (int)x foo: (int)y { // CHECK2-NEXT: rename [[@LINE]]:9 -> [[@LINE]]:12, [[@LINE]]:21 -> [[@LINE]]:24
  [self FOO]; // CHECK1-NEXT: macro [[@LINE]]:9 -> [[@LINE]]:9
}

@end

// RUN: clang-refactor-test rename-initiate -at=%s:13:9 -new-name=bar %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:17:9 -new-name=foo:bar %s | FileCheck --check-prefix=CHECK2 %s

// RUN: not clang-refactor-test rename-initiate -at=%s:1:13 -new-name=foo %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s
// RUN: not clang-refactor-test rename-initiate -at=%s:6:9 -new-name=Bar %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s
// CHECK-ERROR: could not rename symbol at the given location
