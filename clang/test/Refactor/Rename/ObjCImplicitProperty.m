@interface I1

-(int)p1; // CHECK1: rename [[@LINE]]:7 -> [[@LINE]]:9
-(void)setP1:(int)x; // CHECK2: rename [[@LINE]]:8 -> [[@LINE]]:13
-(int)p2; // CHECK3: rename [[@LINE]]:7 -> [[@LINE]]:9

@end

@implementation I1

- (int)p1 { // CHECK1: rename [[@LINE]]:8 -> [[@LINE]]:10
  return 0;
}

- (void)foo: (I1 *)other {
  self.p1 =       // CHECK1: implicit-property [[@LINE]]:8 -> [[@LINE]]:10
                  // CHECK2: implicit-property [[@LINE-1]]:8 -> [[@LINE-1]]:13
             self.p2; // CHECK3: implicit-property [[@LINE]]:19 -> [[@LINE]]:21
  (void)other.p1; // CHECK1: implicit-property [[@LINE]]:15 -> [[@LINE]]:17
                  // CHECK2: implicit-property [[@LINE-1]]:15 -> [[@LINE-1]]:20

  int x = [self p1]; // CHECK1: rename [[@LINE]]:17 -> [[@LINE]]:19
  [self setP1: x];   // CHECK2: rename [[@LINE]]:9 -> [[@LINE]]:14
  x = [other p2];    // CHECK3: rename [[@LINE]]:14 -> [[@LINE]]:16
}

@end

// RUN: clang-refactor-test rename-initiate -at=%s:3:7 -at=%s:19:15 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:4:8 -at=%s:16:8 -new-name=foo %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test rename-initiate -at=%s:5:7 -at=%s:18:19 -new-name=foo %s | FileCheck --check-prefix=CHECK3 %s
