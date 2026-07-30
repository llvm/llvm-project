// RUN: %clang_cc1 -triple x86_64-apple-darwin -fblocks -emit-llvm -o - %s | FileCheck %s

@interface NSObject
+ (id) new;
- (id) init;
@end

@interface Base : NSObject @end

// @implementation Base
// {
//     int dummy;
// }
// @end

@interface Derived : Base
{
    @public int member;
}
@end

@implementation Derived
- (id) init
{
    self = [super init];
    member = 42;
    return self;
}
@end

// CHECK: define internal ptr @"\01-[Derived init]"
// CHECK: [[IVAR:%.*]] = load i64, ptr @"OBJC_IVAR_$_Derived.member", align 8, !invariant.load

void * variant_load_1(int i) {
    void *ptr;
    while (i--) {
        Derived *d = [Derived new];
        ptr = &d->member;
    }
    return ptr;
}

// CHECK-LABEL: define{{.*}} ptr @variant_load_1(i32 noundef %i)
// CHECK: [[IVAR:%.*]] = load i64, ptr @"OBJC_IVAR_$_Derived.member", align 8{{$}}

@interface Container : Derived @end
@implementation Container
- (void *) invariant_load_1
{
    return &self->member;
}
@end

// CHECK-LABEL: define internal ptr @"\01-[Container invariant_load_1]"
// CHECK: [[IVAR:%.*]] = load i64, ptr @"OBJC_IVAR_$_Derived.member", align 8, !invariant.load

@interface ForBlock
{ 
@public
  id foo; 
}
@end

// CHECK-LABEL: define internal ptr @block_block_invoke
// CHECK: load i64, ptr @"OBJC_IVAR_$_ForBlock.foo"
id (^block)(ForBlock*) = ^(ForBlock* a) {
  return a->foo;
};
