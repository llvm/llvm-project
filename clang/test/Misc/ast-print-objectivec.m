// RUN: %clang_cc1 -ast-print %s -o - | FileCheck %s

@interface NSObject @end

@protocol P
- (void)MethP __attribute__((availability(macosx,introduced=10.1.0,deprecated=10.2)));
@end

@interface I : NSObject <P>
- (void)MethI __attribute__((availability(macosx,introduced=10.1.0,deprecated=10.2)));
@end

@interface I(CAT)
- (void)MethCAT __attribute__((availability(macosx,introduced=10_1_0,deprecated=10_2)));
@end

@implementation I
- (void)MethP __attribute__((availability(macosx,introduced=10.1.0,deprecated=10.2))) {}
- (void)MethI __attribute__((availability(macosx,introduced=10.1.0,deprecated=10.2))) {}
@end

// CHECK: @protocol P
// CHECK: - (void) MethP __attribute__((availability(macos, introduced=10.1.0, deprecated=10.2)));
// CHECK: @end

// CHECK: @interface I : NSObject<P> 
// CHECK: - (void) MethI __attribute__((availability(macos, introduced=10.1.0, deprecated=10.2)));
// CHECK: @end

// CHECK: @interface I(CAT)
// CHECK: - (void) MethCAT __attribute__((availability(macos, introduced=10_1_0, deprecated=10_2)));
// CHECK: @end

// CHECK: @implementation I
// CHECK: - (void) MethP __attribute__((availability(macos, introduced=10.1.0, deprecated=10.2))) {
// CHECK: }

// CHECK: - (void) MethI __attribute__((availability(macos, introduced=10.1.0, deprecated=10.2))) {
// CHECK: }

// CHECK: @end

@class C1;
struct __attribute__((objc_bridge_related(C1,,))) S1;

// CHECK: @class C1;
// CHECK: struct __attribute__((objc_bridge_related(C1, , ))) S1;
