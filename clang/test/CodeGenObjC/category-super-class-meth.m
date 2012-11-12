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

// CHECK: define internal i8* @"\01+[Sub2(Category) copy]
// CHECK: [[ONE:%.*]] = load %struct._class_t** @"\01L_OBJC_CLASSLIST_SUP_REFS_$_3"
// CHECK: [[TWO:%.*]] = bitcast %struct._class_t* [[ONE]] to i8*
// CHECK: [[THREE:%.*]] = getelementptr inbounds %struct._objc_super* [[OBJC_SUPER:%.*]], i32 0, i32 1
// CHECK: store i8* [[TWO]], i8** [[THREE]]
// CHECK: [[FOUR:%.*]] = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_"
