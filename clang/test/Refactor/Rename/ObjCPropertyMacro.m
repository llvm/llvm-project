#define IMPLICIT implicit
#define SETIMPLICIT setImplicit

@interface I

- (int)IMPLICIT;
- (void)setImplicit:(int)x; // CHECK1: rename  [[@LINE]]

@end

@implementation I

- (void)foo {
  self.implicit; // CHECK1-NEXT: implicit-property [[@LINE]]
  self.IMPLICIT; // CHECK1-NEXT: implicit-property in macro [[@LINE]]
  self.IMPLICIT = 2; // CHECK1-NEXT: implicit-property in macro [[@LINE]]
}

@end

// RUN: clang-refactor-test rename-initiate -at=%s:7:9 -new-name=bar %s | FileCheck --check-prefix=CHECK1 %s
