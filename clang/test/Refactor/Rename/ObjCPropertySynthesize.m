// XFAIL: *
// TODO: Remove or cut it down to one symbol rename.

@interface SynthesizedIVars

@property int p1; // CHECK1: rename [[@LINE]]:15 -> [[@LINE]]:17
@property(readonly) int p2; // CHECK2PROP: rename [[@LINE]]:25 -> [[@LINE]]:27

@end

@implementation SynthesizedIVars

@synthesize p1 = _p1; // CHECK1: rename [[@LINE]]:13 -> [[@LINE]]:15
                      // CHECK1: rename "_foo" [[@LINE-1]]:18 -> [[@LINE-1]]:21

// The rename of ivar 'p2_' shouldn't initiate the rename of property 'p2'
// because it doesn't follow the default naming convention.
@synthesize p2 = p2_; // CHECK2PROP: rename [[@LINE]]:13 -> [[@LINE]]:15
                      // CHECK2IVAR: rename [[@LINE-1]]:18 -> [[@LINE-1]]:21

- (void)foo:(SynthesizedIVars *)other {
  _p1 =              // CHECK1: rename "_foo" [[@LINE]]:3 -> [[@LINE]]:6
        other->p2_;  // CHECK2IVAR: rename [[@LINE]]:16 -> [[@LINE]]:19
  other->p2_ =       // CHECK2IVAR: rename [[@LINE]]:10 -> [[@LINE]]:13
               _p1;  // CHECK1: rename "_foo" [[@LINE]]:16 -> [[@LINE]]:19
  self.p1 =          // CHECK1: rename [[@LINE]]:8 -> [[@LINE]]:10
           other.p2; // CHECK2PROP: rename [[@LINE]]:18 -> [[@LINE]]:20
}

@end

// RUN: clang-refactor-test rename-initiate -at=%s:3:15 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:10:13 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:10:18 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:19:3 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:22:16 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:23:8 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s

// RUN: clang-refactor-test rename-initiate -at=%s:4:25 -new-name=foo %s | FileCheck --check-prefix=CHECK2PROP %s
// RUN: clang-refactor-test rename-initiate -at=%s:15:13 -new-name=foo %s | FileCheck --check-prefix=CHECK2PROP %s
// RUN: clang-refactor-test rename-initiate -at=%s:24:18 -new-name=foo %s | FileCheck --check-prefix=CHECK2PROP %s

// RUN: clang-refactor-test rename-initiate -at=%s:15:18 -new-name=foo %s | FileCheck --check-prefix=CHECK2IVAR %s
// RUN: clang-refactor-test rename-initiate -at=%s:20:16 -new-name=foo %s | FileCheck --check-prefix=CHECK2IVAR %s
// RUN: clang-refactor-test rename-initiate -at=%s:21:10 -new-name=foo %s | FileCheck --check-prefix=CHECK2IVAR %s

@interface SynthesizedExplicitIVars {
  int _p3; // CHECK3: rename "_foo" [[@LINE]]:7 -> [[@LINE]]:10
}

@property int p3; // CHECK3: rename [[@LINE]]:15 -> [[@LINE]]:17
@property int p4; // CHECK4: rename [[@LINE]]:15 -> [[@LINE]]:17

@end

@implementation SynthesizedExplicitIVars {
  int _p4; // CHECK4: rename "_foo" [[@LINE]]:7 -> [[@LINE]]:10
}

@synthesize p3 = _p3; // CHECK3: rename [[@LINE]]:13 -> [[@LINE]]:15
                      // CHECK3: rename "_foo" [[@LINE-1]]:18 -> [[@LINE-1]]:21
@synthesize p4 = _p4; // CHECK4: rename [[@LINE]]:13 -> [[@LINE]]:15
                      // CHECK4: rename "_foo" [[@LINE-1]]:18 -> [[@LINE-1]]:21

@end

// RUN: clang-refactor-test rename-initiate -at=%s:45:7 -new-name=foo %s | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test rename-initiate -at=%s:48:15 -new-name=foo %s | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test rename-initiate -at=%s:57:13 -new-name=foo %s | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test rename-initiate -at=%s:57:18 -new-name=foo %s | FileCheck --check-prefix=CHECK3 %s

// RUN: clang-refactor-test rename-initiate -at=%s:49:15 -new-name=foo %s | FileCheck --check-prefix=CHECK4 %s
// RUN: clang-refactor-test rename-initiate -at=%s:54:7 -new-name=foo %s | FileCheck --check-prefix=CHECK4 %s
// RUN: clang-refactor-test rename-initiate -at=%s:59:13 -new-name=foo %s | FileCheck --check-prefix=CHECK4 %s
// RUN: clang-refactor-test rename-initiate -at=%s:59:18 -new-name=foo %s | FileCheck --check-prefix=CHECK4 %s

@interface SynthesizedWithoutIVarName

@property int p5; // CHECK5: rename [[@LINE]]:15 -> [[@LINE]]:17

@end

@implementation SynthesizedWithoutIVarName {
  int _p5; // CHECK5-NOT: rename "" [[@LINE]]
           // CHECK5-NOT: rename [[@LINE-1]]
  int p5;  // CHECK5: rename [[@LINE]]:7 -> [[@LINE]]:9
}

@synthesize p5; // CHECK5: rename [[@LINE]]:13 -> [[@LINE]]:15

@end

// RUN: clang-refactor-test rename-initiate -at=%s:76:15 -new-name=foo %s | FileCheck --check-prefix=CHECK5 %s
// RUN: clang-refactor-test rename-initiate -at=%s:83:7 -new-name=foo %s | FileCheck --check-prefix=CHECK5 %s
// RUN: clang-refactor-test rename-initiate -at=%s:86:13 -new-name=foo %s | FileCheck --check-prefix=CHECK5 %s

@interface A
@property int p6;
@end

@interface C : A
@end

@implementation C

@synthesize p6;
// CHECK6: Renaming 4 symbols
// CHECK6-NEXT: 'c:objc(cs)A(py)p6'

@end

// RUN: clang-refactor-test rename-initiate -at=%s:103:13 -new-name=foo -dump-symbols %s | FileCheck --check-prefix=CHECK6 %s
