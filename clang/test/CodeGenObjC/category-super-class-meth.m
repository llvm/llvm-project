// RUN: %clang_cc1 %s -emit-llvm -triple x86_64-apple-darwin -o - | FileCheck %s
// rdar://12459358
@interface NSObject 
-(id)copy;
+(id)copy;
@end

@interface Sub1 : NSObject @end

@implementation Sub1
-(id)copy { return [super copy]; }  // ok: instance method in class
+(id)copy { return [super copy]; }  // ok: class method in class
@end

@interface Sub2 : NSObject @end

@interface Sub2 (Category) @end

@implementation Sub2 (Category)
-(id)copy { return [super copy]; }  // ok: instance method in category
+(id)copy { return [super copy]; }  // BAD: class method in category
@end

// CHECK: define internal ptr @"\01+[Sub2(Category) copy]
// CHECK: [[ONE:%.*]] = load ptr, ptr @"OBJC_CLASSLIST_SUP_REFS_$_.3"
// CHECK: [[THREE:%.*]] = getelementptr inbounds %struct._objc_super, ptr [[OBJC_SUPER:%.*]], i32 0, i32 1
// CHECK: store ptr [[ONE]], ptr [[THREE]]
// CHECK: [[FOUR:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_
