@interface DynamicProperty

@property int p1; // CHECK1: rename [[@LINE]]:15 -> [[@LINE]]:17
@property(readonly) int p2; // CHECK2: rename [[@LINE]]:25 -> [[@LINE]]:27

@end

@implementation DynamicProperty

@dynamic p1; // CHECK1: rename [[@LINE]]:10 -> [[@LINE]]:12

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

- (void)foo:(DynamicProperty *)other {
  self.p1 = // CHECK1: rename [[@LINE]]:8 -> [[@LINE]]:10
            other.p2; // CHECK2: rename [[@LINE]]:19 -> [[@LINE]]:21
}

@end

// RUN: clang-refactor-test rename-initiate -at=%s:3:15 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:10:10 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:14:8 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:18:9 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:26:8 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s

// RUN: clang-refactor-test rename-initiate -at=%s:4:25 -new-name=foo %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test rename-initiate -at=%s:12:10 -new-name=foo %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test rename-initiate -at=%s:21:8 -new-name=foo %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test rename-initiate -at=%s:27:19 -new-name=foo %s | FileCheck --check-prefix=CHECK2 %s
