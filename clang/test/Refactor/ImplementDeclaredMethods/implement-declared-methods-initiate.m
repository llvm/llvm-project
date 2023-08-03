
@protocol P

- (void)method;

@end

@interface MyClass {
  int ivar;
}

@property int prop;

- (void)method;

// comment
+ (void)classMethod;

- (void)implementedMethod;

- (void)method:(int)x with:(int)y;

@end

@implementation MyClass

- (void)implementedMethod {

}

@end


// CHECK1: Initiated the 'implement-declared-methods' action at [[@LINE-20]]:1
// CHECK2: Initiated the 'implement-declared-methods' action at [[@LINE-18]]:1
// CHECK3: Initiated the 'implement-declared-methods' action at [[@LINE-15]]:1

// RUN: clang-refactor-test initiate -action implement-declared-methods -in=%s:14:1-14 %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test initiate -action implement-declared-methods -in=%s:17:1-20 %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test initiate -action implement-declared-methods -in=%s:21:1-34 %s | FileCheck --check-prefix=CHECK3 %s

// RUN: not clang-refactor-test initiate -action implement-declared-methods -in=%s:4:1-end -in=%s:27:1-end -in=%s:8:1-end -in=%s:9:1-end -in=%s:12:1-end -in=%s:16:1-end -in=%s:19:1-end -in=%s:23:1-end %s  2>&1 | FileCheck --check-prefix=CHECK-NO %s

// CHECK-NO: Failed to initiate the refactoring action

// method, classMethod, method:with: :
// CHECK4: Initiated the 'implement-declared-methods' action at [[@LINE-33]]:1 -> [[@LINE-26]]:35
// RUN: clang-refactor-test initiate -action implement-declared-methods -selected=%s:8:1-24:3 -selected=%s:9:1-23:3 -selected=%s:14:1-21:35 -selected=%s:14:14-21:2  %s | FileCheck --check-prefix=CHECK4 %s

// classMethod, method:with:
// CHECK5: Initiated the 'implement-declared-methods' action at [[@LINE-34]]:1 -> [[@LINE-30]]:35
// RUN: clang-refactor-test initiate -action implement-declared-methods -selected=%s:17:1-21:35 -selected=%s:16:1-22:1 -selected=%s:17:20-21:2 %s | FileCheck --check-prefix=CHECK5 %s

// classMethod
// CHECK6: Initiated the 'implement-declared-methods' action at [[@LINE-38]]:1
// RUN: clang-refactor-test initiate -action implement-declared-methods -selected=%s:17:1-17:10 -selected=%s:17:20-18:1 %s | FileCheck --check-prefix=CHECK6 %s

// RUN: not clang-refactor-test initiate -action implement-declared-methods -selected=%s:2:1-30:10 -selected=%s:2:1-6:10 -selected=%s:6:1-25:2 -selected=%s:27:1-29:1 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s

// Methods declared in class extensions / categories should be supported:

@interface I2

@end

@interface I2 ()

- (void)method;
+ (void)classMethod;
- (void)implementedMethod;

@end
// CHECK7: Initiated the 'implement-declared-methods' action at [[@LINE-5]]:1
// RUN: clang-refactor-test initiate -action implement-declared-methods -at=%s:68:1 %s | FileCheck --check-prefix=CHECK7 %s
// RUN: not clang-refactor-test initiate -action implement-declared-methods -at=%s:70:1 %s  2>&1 | FileCheck --check-prefix=CHECK-NO %s

// CHECK8: Initiated the 'implement-declared-methods' action at [[@LINE-9]]:1 -> [[@LINE-8]]:21
// RUN: clang-refactor-test initiate -action implement-declared-methods -selected=%s:68:1-71:1 %s | FileCheck --check-prefix=CHECK8 %s

@implementation I2

- (void)implementedMethod {
}

@end

@interface I2 (Extension)

- (void)methodExt;
+ (void)classMethodExt;
- (void)implementedMethodExt;

@end
// CHECK9: Initiated the 'implement-declared-methods' action at [[@LINE-5]]:1
// RUN: clang-refactor-test initiate -action implement-declared-methods -at=%s:89:1 %s | FileCheck --check-prefix=CHECK9 %s
// RUN: not clang-refactor-test initiate -action implement-declared-methods -at=%s:91:1 %s  2>&1 | FileCheck --check-prefix=CHECK-NO %s

// CHECK10: Initiated the 'implement-declared-methods' action at [[@LINE-9]]:1 -> [[@LINE-8]]:24
// RUN: clang-refactor-test initiate -action implement-declared-methods -selected=%s:89:1-92:1 %s | FileCheck --check-prefix=CHECK10 %s

@implementation I2 (Extension)

- (void)implementedMethodExt {
}

@end
