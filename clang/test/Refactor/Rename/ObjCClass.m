@class I1, // CHECK1: rename [[@LINE]]:8 -> [[@LINE]]:10
       I2; // CHECK2: rename [[@LINE]]:8 -> [[@LINE]]:10

@interface I1 // CHECK1: rename [[@LINE]]:12 -> [[@LINE]]:14
@end

// RUN: clang-refactor-test rename-initiate -at=%s:1:8 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:4:12 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:2:8 -new-name=foo %s | FileCheck --check-prefix=CHECK2 %s

@implementation I1 { // CHECK1: rename [[@LINE]]:17 -> [[@LINE]]:19
  I1 *interfaceIVar; // CHECK1: rename [[@LINE]]:3 -> [[@LINE]]:5
                     // CHECK4: rename [[@LINE-1]]:7 -> [[@LINE-1]]:20
  int ivar; // CHECK3: rename [[@LINE]]:7 -> [[@LINE]]:11
}

// RUN: clang-refactor-test rename-initiate -at=%s:11:17 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:12:3 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:21:20 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s

-(void)foo: (const I1 *)bar { // CHECK1: rename [[@LINE]]:20 -> [[@LINE]]:22

  ivar = 1;        // CHECK3: rename [[@LINE]]:3 -> [[@LINE]]:7
  self->ivar = 2;  // CHECK3: rename [[@LINE]]:9 -> [[@LINE]]:13
  print(bar->ivar);// CHECK3: rename [[@LINE]]:14 -> [[@LINE]]:18
  interfaceIVar->ivar = 4; // CHECK4: rename [[@LINE]]:3 -> [[@LINE]]:16
                           // CHECK3: rename [[@LINE-1]]:18 -> [[@LINE-1]]:22
}

// RUN: clang-refactor-test rename-initiate -at=%s:14:7 -new-name=foo %s | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test rename-initiate -at=%s:23:3 -new-name=foo %s | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test rename-initiate -at=%s:24:9 -new-name=foo %s | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test rename-initiate -at=%s:25:14 -new-name=foo %s | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test rename-initiate -at=%s:26:18 -new-name=foo %s | FileCheck --check-prefix=CHECK3 %s

// RUN: clang-refactor-test rename-initiate -at=%s:12:7 -new-name=foo %s | FileCheck --check-prefix=CHECK4 %s
// RUN: clang-refactor-test rename-initiate -at=%s:26:3 -new-name=foo %s | FileCheck --check-prefix=CHECK4 %s

@end

@interface I1 (Category) // CHECK1: rename [[@LINE]]:12 -> [[@LINE]]:14
@end                     // CHECK5: rename [[@LINE-1]]:16 -> [[@LINE-1]]:24

@implementation I1 (Category) // CHECK1: rename [[@LINE]]:17 -> [[@LINE]]:19
@end                          // CHECK5: rename [[@LINE-1]]:21 -> [[@LINE-1]]:29

// RUN: clang-refactor-test rename-initiate -at=%s:41:12 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:44:17 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s

// RUN: clang-refactor-test rename-initiate -at=%s:41:16 -new-name=foo %s | FileCheck --check-prefix=CHECK5 %s
// RUN: clang-refactor-test rename-initiate -at=%s:44:21 -new-name=foo %s | FileCheck --check-prefix=CHECK5 %s

// Implementation only-category:

@interface I3 // CHECK6: rename [[@LINE]]:12 -> [[@LINE]]:14
@end

@implementation I3 (DummyCategory) // CHECK6: rename [[@LINE]]:17 -> [[@LINE]]:19
@end                               // CHECK7: rename [[@LINE-1]]:21 -> [[@LINE-1]]:34

// RUN: clang-refactor-test rename-initiate -at=%s:55:12 -new-name=foo %s | FileCheck --check-prefix=CHECK6 %s
// RUN: clang-refactor-test rename-initiate -at=%s:58:17 -new-name=foo %s | FileCheck --check-prefix=CHECK6 %s

// RUN: clang-refactor-test rename-initiate -at=%s:58:21 -new-name=foo %s | FileCheck --check-prefix=CHECK7 %s

// Class extension:

@interface I3 () // CHECK6: rename [[@LINE]]:12 -> [[@LINE]]:14
@end

@implementation I3 // CHECK6: rename [[@LINE]]:17 -> [[@LINE]]:19
@end

// RUN: clang-refactor-test rename-initiate -at=%s:68:12 -new-name=foo %s | FileCheck --check-prefix=CHECK6 %s
// RUN: clang-refactor-test rename-initiate -at=%s:71:17 -new-name=foo %s | FileCheck --check-prefix=CHECK6 %s

// Ivar declared in the interface:

@interface I4 {
  @public
  int ivar1; // CHECK8: rename [[@LINE]]:7 -> [[@LINE]]:12
}
@end

@implementation I4 {
}

- (void)foo {
  ivar1 = 0; // CHECK8: rename [[@LINE]]:3 -> [[@LINE]]:8
}

@end

// RUN: clang-refactor-test rename-initiate -at=%s:81:7 -new-name=foo %s | FileCheck --check-prefix=CHECK8 %s
// RUN: clang-refactor-test rename-initiate -at=%s:89:3 -new-name=foo %s | FileCheck --check-prefix=CHECK8 %s
