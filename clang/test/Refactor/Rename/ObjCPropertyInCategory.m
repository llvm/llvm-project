@interface I1

@end

@interface I1 (Category)

@property int p1; // CHECK1: rename [[@LINE]]:15 -> [[@LINE]]:17
@property(readonly) int p2; // CHECK2: rename [[@LINE]]:25 -> [[@LINE]]:27

- (int)p1; // CHECK1: rename [[@LINE]]:8 -> [[@LINE]]:10

@end

@implementation I1 (Category)

@dynamic p2; // CHECK2: rename [[@LINE]]:10 -> [[@LINE]]:12

- (int)p1 { // CHECK1: rename [[@LINE]]:8 -> [[@LINE]]:10
  return 1;
}
// TODO: Remove
- (void)setP1:(int)x { // CHECK1: rename [[@LINE]]:9 -> [[@LINE]]:14
}

- (int)p2 { // CHECK2: rename [[@LINE]]:8 -> [[@LINE]]:10
  return 2;
}

- (void)foo:(I1 *)other {
  self.p1 = // CHECK1: rename [[@LINE]]:8 -> [[@LINE]]:10
            other.p2; // CHECK2: rename [[@LINE]]:19 -> [[@LINE]]:21
}

@end

// RUN: clang-refactor-test rename-initiate -at=%s:7:15 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:10:8 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:18:8 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:22:9 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:30:8 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s

// RUN: clang-refactor-test rename-initiate -at=%s:8:25 -new-name=foo %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test rename-initiate -at=%s:16:10 -new-name=foo %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test rename-initiate -at=%s:25:8 -new-name=foo %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test rename-initiate -at=%s:31:19 -new-name=foo %s | FileCheck --check-prefix=CHECK2 %s
