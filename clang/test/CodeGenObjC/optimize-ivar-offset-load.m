// RUN: %clang_cc1 -triple x86_64-apple-darwin10  -O0 -emit-llvm %s -o -  | FileCheck %s
// rdar://16095748

@interface MyNSObject 
@end

@interface SampleClass : MyNSObject {
    @public
    int _value;
}
+ (SampleClass*) new;
@end

@interface AppDelegate  : MyNSObject
@end

extern void foo(int);

@implementation AppDelegate
- (void)application
{
    // Create set of objects in loop
    for(int i = 0; i < 2; i++) {
        SampleClass *sample = [SampleClass new];
        foo (sample->_value);
    }
}
@end
// CHECK: [[IVAR:%.*]]  = load i64, i64* @"OBJC_IVAR_$_SampleClass._value", align 8
// CHECK: [[THREE:%.*]] = bitcast [[ONE:%.*]]* [[CALL:%.*]] to i8*
// CHECK: [[ADDPTR:%.*]] = getelementptr inbounds i8, i8* [[THREE]], i64 [[IVAR]]
// CHECK: [[FOUR:%.*]] = bitcast i8* [[ADDPTR]] to i32*
// CHECK: [[FIVE:%.*]] = load i32, i32* [[FOUR]], align 4
// CHECK:   call void @foo(i32 noundef [[FIVE]])

@implementation SampleClass
+ (SampleClass*) new { return 0; }
- (void) SampleClassApplication
{
    // Create set of objects in loop
    for(int i = 0; i < 2; i++) {
        SampleClass *sample = [SampleClass new];
        foo (sample->_value);
    }
}
@end
// CHECK: [[ZERO:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_, align 8, !invariant.load
// CHECK: [[IVAR:%.*]] = load i64, i64* @"OBJC_IVAR_$_SampleClass._value", align 8, !invariant.load

@interface Sample : SampleClass @end

@implementation Sample
- (void) SampleApplication
{
    // Create set of objects in loop
    for(int i = 0; i < 2; i++) {
        SampleClass *sample = [SampleClass new];
        foo (sample->_value);
    }
}
@end
// CHECK: [[ZERO:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_, align 8, !invariant.load 
// CHECK: [[IVAR:%.*]] = load i64, i64* @"OBJC_IVAR_$_SampleClass._value", align 8, !invariant.load

