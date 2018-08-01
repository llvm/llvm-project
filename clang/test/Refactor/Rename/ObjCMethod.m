@interface Test

- (void)foo; // CHECK1: rename [[@LINE]]:9 -> [[@LINE]]:12
- (int)performAction:(int)action with:(int)value; // CHECK2: rename [[@LINE]]:8 -> [[@LINE]]:21, [[@LINE]]:34 -> [[@LINE]]:38

@end

@implementation Test

- (void)foo { // CHECK1: rename [[@LINE]]:9 -> [[@LINE]]:12
}

- (int)performAction:(int)action with:(int)value { // CHECK2: rename [[@LINE]]:8 -> [[@LINE]]:21, [[@LINE]]:34 -> [[@LINE]]:38
    return action + value;
}

+ (void)foo:(Test*)t {      // CHECK1-NOT: rename [[@LINE]]
    [t foo];                // CHECK1: rename [[@LINE]]:8 -> [[@LINE]]:11
    SEL s = @selector(foo);
    [Test foo:t];           // CHECK1-NOT: rename [[@LINE]]
    [t performAction: 2 with: 4]; // CHECK2: rename [[@LINE]]:8 -> [[@LINE]]:21, [[@LINE]]:25 -> [[@LINE]]:29
    SEL s1 = @selector(foo:);
    SEL s2 = @selector(performAction:
                       with:);
    SEL s3 = @selector(performAction:);
    SEL s4 = @selector(performAction);
}

@end

// RUN: clang-refactor-test rename-initiate -at=%s:3:9 -new-name=bar %s -Wno-objc-root-class | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:10:9 -new-name=bar %s -Wno-objc-root-class | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:18:8 -new-name=bar %s -Wno-objc-root-class | FileCheck --check-prefix=CHECK1 %s

// RUN: clang-refactor-test rename-initiate -at=%s:4:8 -new-name=doSomething:to %s -Wno-objc-root-class | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test rename-initiate -at=%s:13:8 -new-name=doSomething:to: %s -Wno-objc-root-class | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test rename-initiate -at=%s:21:8 -new-name=doSomething:to: %s -Wno-objc-root-class | FileCheck --check-prefix=CHECK2 %s




@interface SuperClass

- (void)foo; // CHECK-OVERRIDEFOO: rename [[@LINE]]:9 -> [[@LINE]]:12
- (int)compareTo:(SuperClass *)other with:(int)options; // CHECK-OVERRIDECOMP: rename [[@LINE]]:8 -> [[@LINE]]:17, [[@LINE]]:38 -> [[@LINE]]:42

@end

@implementation SuperClass

- (void)foo { // CHECK-OVERRIDEFOO: rename [[@LINE]]:9 -> [[@LINE]]:12
  return;
}

- (int)compareTo:(SuperClass *)other with:(int)options { // CHECK-OVERRIDECOMP: rename [[@LINE]]:8 -> [[@LINE]]:17, [[@LINE]]:38 -> [[@LINE]]:42
  return 0;
}

@end

@interface SubClass : SuperClass

- (void)foo; // CHECK-OVERRIDEFOO: rename [[@LINE]]:9 -> [[@LINE]]:12

@end

@implementation SubClass

- (void)foo { // CHECK-OVERRIDEFOO: rename [[@LINE]]:9 -> [[@LINE]]:12
  [super foo]; // CHECK-OVERRIDEFOO: rename [[@LINE]]:10 -> [[@LINE]]:13
}

@end

@interface SubClassTheSecond : SubClass

- (void)foo; // CHECK-OVERRIDEFOO: rename [[@LINE]]:9 -> [[@LINE]]:12
- (int)compareTo:(SuperClass *)other with:(int)options; // CHECK-OVERRIDECOMP: rename [[@LINE]]:8 -> [[@LINE]]:17, [[@LINE]]:38 -> [[@LINE]]:42

@end

@implementation SubClassTheSecond

- (void)foo { // CHECK-OVERRIDEFOO: rename [[@LINE]]:9 -> [[@LINE]]:12
  return;
}
- (int)compareTo:(SuperClass *)other // CHECK-OVERRIDECOMP: rename [[@LINE]]:8 -> [[@LINE]]:17, [[@LINE+1]]:8 -> [[@LINE+1]]:12
       with:(int)options {
  [other foo]; // CHECK-OVERRIDEFOO: rename [[@LINE]]:10 -> [[@LINE]]:13
  return [super compareTo: other with: options]; // CHECK-OVERRIDECOMP: rename [[@LINE]]:17 -> [[@LINE]]:26, [[@LINE]]:34 -> [[@LINE]]:38
}

@end

@interface UnrelatedClass

- (void)foo; // CHECK-OVERRIDEFOO-NOT: rename [[@LINE]]

@end

@interface UnrelatedSubClass : UnrelatedClass
// This method doesn't override SuperClass.foo, so verify that this occurrence
// isn't renamed even though its selector is the same.
- (void)foo; // CHECK-OVERRIDEFOO-NOT: rename [[@LINE]]

@end

// RUN: clang-refactor-test rename-initiate -at=%s:44:9 -new-name=bar %s -Wno-objc-root-class | FileCheck --check-prefix=CHECK-OVERRIDEFOO %s
// RUN: clang-refactor-test rename-initiate -at=%s:51:9 -new-name=bar %s -Wno-objc-root-class | FileCheck --check-prefix=CHECK-OVERRIDEFOO %s
// RUN: clang-refactor-test rename-initiate -at=%s:63:9 -new-name=bar %s -Wno-objc-root-class | FileCheck --check-prefix=CHECK-OVERRIDEFOO %s
// RUN: clang-refactor-test rename-initiate -at=%s:69:9 -new-name=bar %s -Wno-objc-root-class | FileCheck --check-prefix=CHECK-OVERRIDEFOO %s
// RUN: clang-refactor-test rename-initiate -at=%s:70:10 -new-name=bar %s -Wno-objc-root-class | FileCheck --check-prefix=CHECK-OVERRIDEFOO %s
// RUN: clang-refactor-test rename-initiate -at=%s:77:9 -new-name=bar %s -Wno-objc-root-class | FileCheck --check-prefix=CHECK-OVERRIDEFOO %s
// RUN: clang-refactor-test rename-initiate -at=%s:84:9 -new-name=bar %s -Wno-objc-root-class | FileCheck --check-prefix=CHECK-OVERRIDEFOO %s
// RUN: clang-refactor-test rename-initiate -at=%s:89:10 -new-name=bar %s -Wno-objc-root-class | FileCheck --check-prefix=CHECK-OVERRIDEFOO %s

// RUN: clang-refactor-test rename-initiate -at=%s:45:8 -new-name=a:b: %s -Wno-objc-root-class | FileCheck --check-prefix=CHECK-OVERRIDECOMP %s
// RUN: clang-refactor-test rename-initiate -at=%s:55:8 -new-name=a:b: %s -Wno-objc-root-class | FileCheck --check-prefix=CHECK-OVERRIDECOMP %s
// RUN: clang-refactor-test rename-initiate -at=%s:78:8 -new-name=a:b %s -Wno-objc-root-class | FileCheck --check-prefix=CHECK-OVERRIDECOMP %s
// RUN: clang-refactor-test rename-initiate -at=%s:87:9 -new-name=a:b: %s -Wno-objc-root-class | FileCheck --check-prefix=CHECK-OVERRIDECOMP %s
// RUN: clang-refactor-test rename-initiate -at=%s:90:17 -new-name=a:b %s -Wno-objc-root-class | FileCheck --check-prefix=CHECK-OVERRIDECOMP %s
// RUN: clang-refactor-test rename-initiate -at=%s:90:34 -new-name=a:b %s -Wno-objc-root-class | FileCheck --check-prefix=CHECK-OVERRIDECOMP %s

// Don't allow implicit parameters:
@interface Foo
- (void)foo;
@end

@implementation Foo
- (void)foo {
  self = 0;
}
@end
// RUN: not clang-refactor-test rename-initiate -at=%s:130:3 -new-name=foo %s -Wno-objc-root-class 2>&1 | FileCheck --check-prefix=CHECK-NORENAME %s
// CHECK-NORENAME: could not rename symbol at the given location

@interface EmptySelectorsRule_Psych

- (void):(int)_ :(int) m:(int)z; // EMPTY-SELECTOR: rename [[@LINE]]:9 -> [[@LINE]]:9, [[@LINE]]:17 -> [[@LINE]]:17, [[@LINE]]:25 -> [[@LINE]]:25

@end

@implementation EmptySelectorsRule_Psych

- (void) :(int)_ :(int)m :(int)z { // EMPTY-SELECTOR: rename [[@LINE]]:10 -> [[@LINE]]:10, [[@LINE]]:18 -> [[@LINE]]:18, [[@LINE]]:26 -> [[@LINE]]:26
    [self: 15:0 :3]; // EMPTY-SELECTOR: rename [[@LINE]]:10 -> [[@LINE]]:10, [[@LINE]]:14 -> [[@LINE]]:14, [[@LINE]]:17 -> [[@LINE]]:17
}
// RUN: clang-refactor-test rename-initiate -at=%s:139:9 -new-name=test:a:: %s -Wno-objc-root-class | FileCheck --check-prefix=EMPTY-SELECTOR %s

@end
