@interface ExplicitClassProperty

@property(class) int p1; // CHECK1: rename [[@LINE]]:22 -> [[@LINE]]:24
@property(class, readonly) int p2; // CHECK2: rename [[@LINE]]:32 -> [[@LINE]]:34

+ (int)p1; // CHECK1: rename [[@LINE]]:8 -> [[@LINE]]:10

@end

@implementation ExplicitClassProperty

@dynamic p1; // CHECK1: rename [[@LINE]]:10 -> [[@LINE]]:12

@dynamic p2; // CHECK2: rename [[@LINE]]:10 -> [[@LINE]]:12

+ (int)p1 { // CHECK1: rename [[@LINE]]:8 -> [[@LINE]]:10
  return 1;
}
// TODO: remove
+ (void)setP1:(int)x { // CHECK1: rename [[@LINE]]:9 -> [[@LINE]]:14
}

+ (int)p2 { // CHECK2: rename [[@LINE]]:8 -> [[@LINE]]:10
  return 2;
}

- (void)foo {
  ExplicitClassProperty.p1 =  // CHECK1: rename [[@LINE]]:25 -> [[@LINE]]:27
    ExplicitClassProperty.p2; // CHECK2: rename [[@LINE]]:27 -> [[@LINE]]:29
}

@end

// RUN: clang-refactor-test rename-initiate -at=%s:3:22 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:6:8 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:12:10 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:16:8 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:20:9 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:28:25 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s

// RUN: clang-refactor-test rename-initiate -at=%s:4:32 -new-name=foo %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test rename-initiate -at=%s:14:10 -new-name=foo %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test rename-initiate -at=%s:23:8 -new-name=foo %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test rename-initiate -at=%s:29:27 -new-name=foo %s | FileCheck --check-prefix=CHECK2 %s

@interface ImplicitClassProperty

+(int)p3; // CHECK3: rename [[@LINE]]:7 -> [[@LINE]]:9
+(void)setP3:(int)x; // CHECK4: rename [[@LINE]]:8 -> [[@LINE]]:13
+(int)p4; // CHECK5: rename [[@LINE]]:7 -> [[@LINE]]:9

@end

@implementation ImplicitClassProperty

+ (int)p3 { // CHECK3: rename [[@LINE]]:8 -> [[@LINE]]:10
  return 0;
}

- (void)foo {
  ImplicitClassProperty.p3 =      // CHECK3: implicit-property [[@LINE]]:25 -> [[@LINE]]:27
                                  // CHECK4: implicit-property [[@LINE-1]]:25 -> [[@LINE-1]]:30
        ImplicitClassProperty.p4; // CHECK5: implicit-property [[@LINE]]:31 -> [[@LINE]]:33
  (void)ImplicitClassProperty.p3; // CHECK3: implicit-property [[@LINE]]:31 -> [[@LINE]]:33
                                  // CHECK4: implicit-property [[@LINE-1]]:31 -> [[@LINE-1]]:36

  int x = [ImplicitClassProperty p3]; // CHECK3: rename [[@LINE]]:34 -> [[@LINE]]:36
  [ImplicitClassProperty setP3: x];   // CHECK4: rename [[@LINE]]:26 -> [[@LINE]]:31
  x = [ImplicitClassProperty p4];     // CHECK5: rename [[@LINE]]:30 -> [[@LINE]]:32
}

@end

// RUN: clang-refactor-test rename-initiate -at=%s:48:7 -at=%s:64:31 -new-name=foo %s | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test rename-initiate -at=%s:49:8 -at=%s:61:25 -new-name=foo %s | FileCheck --check-prefix=CHECK4 %s
// RUN: clang-refactor-test rename-initiate -at=%s:50:7 -at=%s:63:31 -new-name=foo %s | FileCheck --check-prefix=CHECK5 %s








@interface ClassReceivers // CHECK-RECEIVER: rename [[@LINE]]:12 -> [[@LINE]]:26

@property(class) int p1;
+ (int)implicit;
+ (void)setImplicit:(int)x;

@end

void classReceivers() {
  ClassReceivers.p1 = 0; // CHECK-RECEIVER: rename [[@LINE]]:3 -> [[@LINE]]:17
  int y = ClassReceivers.p1; // CHECK-RECEIVER: rename [[@LINE]]:11 -> [[@LINE]]:25
  ClassReceivers.implicit = 0; // CHECK-RECEIVER: rename [[@LINE]]:3 -> [[@LINE]]:17
  int x = ClassReceivers.implicit; // CHECK-RECEIVER: rename [[@LINE]]:11 -> [[@LINE]]:25
}

// RUN: clang-refactor-test rename-initiate -at=%s:94:3 -at=%s:95:11 -at=%s:96:3 -at=%s:97:11 -new-name=x %s | FileCheck --check-prefix=CHECK-RECEIVER %s
