@class I1, // CHECK1: rename [[@LINE]]:8 -> [[@LINE]]:10
       I2; // CHECK2: rename [[@LINE]]:8 -> [[@LINE]]:10

@interface I1 // CHECK1: rename [[@LINE]]:12 -> [[@LINE]]:14
@end

@compatibility_alias I1Alias I1; // CHECK3: rename [[@LINE]]:22 -> [[@LINE]]:29
                                 // CHECK1: rename [[@LINE-1]]:30 -> [[@LINE-1]]:32

@compatibility_alias I2Alias I2; // CHECK4: rename [[@LINE]]:22 -> [[@LINE]]:29
                                 // CHECK2: rename [[@LINE-1]]:30 -> [[@LINE-1]]:32

// RUN: clang-refactor-test rename-initiate -at=%s:7:30 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:10:30 -new-name=foo %s | FileCheck --check-prefix=CHECK2 %s

// RUN: clang-refactor-test rename-initiate -at=%s:7:22 -new-name=foo %s | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test rename-initiate -at=%s:10:22 -new-name=foo %s | FileCheck --check-prefix=CHECK4 %s

// TODO: Implement TypeLocs for @compatibility_alias (rdar://29245831)
// XFAIL: *
void foo(I1Alias *object) { // CHECK3: rename [[@LINE]]:10 -> [[@LINE]]:17
}

@implementation I1 { // CHECK1: rename [[@LINE]]:17 -> [[@LINE]]:19
  I1Alias *object;   // CHECK3: rename [[@LINE]]:3 -> [[@LINE]]:10
}

-(const I1Alias *)foo:(I2Alias *)object { // CHECK3: rename [[@LINE]]:9 -> [[@LINE]]:16
                                          // CHECK4: rename [[@LINE-1]]:24 -> [[@LINE-1]]:31
  return (const I1Alias *)self->object;   // CHECK3: rename [[@LINE]]:17 -> [[@LINE]]:24
}

@end

@interface I3: I1Alias // CHECK3: rename [[@LINE]]:16 -> [[@LINE]]:23
@end

// RUN: clang-refactor-test rename-initiate -at=%s:21:10 -new-name=foo %s | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test rename-initiate -at=%s:25:3 -new-name=foo %s | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test rename-initiate -at=%s:28:9 -new-name=foo %s | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test rename-initiate -at=%s:30:17 -new-name=foo %s | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test rename-initiate -at=%s:35:16 -new-name=foo %s | FileCheck --check-prefix=CHECK3 %s

// RUN: clang-refactor-test rename-initiate -at=%s:28:24 -new-name=foo %s | FileCheck --check-prefix=CHECK4 %s
