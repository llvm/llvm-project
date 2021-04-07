// RUN: %clang_cc1 -verify -fobjc-disable-direct-methods-for-testing %s

// expected-no-diagnostics

#define DIRECT __attribute__((objc_direct))
#define DIRECT_MEMBERS __attribute__((objc_direct_members))

__attribute__((objc_root_class))
@interface X
-(void)direct_method DIRECT;
@end

@implementation X
-(void)direct_method DIRECT {}
@end

__attribute__((objc_root_class))
DIRECT_MEMBERS
@interface Y
-(void)direct_method2;
@end

@implementation Y
-(void)direct_method2 {}
@end

__attribute__((objc_root_class))
@interface Z
@property (direct) int direct_property;
@end

@implementation Z @end
