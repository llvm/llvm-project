// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s
// rdar://13598026

@interface NSObject @end

@class NSString;

int __CFConstantStringClassReference[24];

@interface Bar : NSObject
+(void)format:(NSString *)format,...;
@end

@interface Foo : NSObject
@end


static inline void _inlineFunction() {
    [Bar format:@" "];
}

@implementation Foo


+(NSString *)someMethod {
   return @"";
}

-(void)someMethod {
   _inlineFunction();
}
@end

// CHECK: @__CFConstantStringClassReference ={{.*}} global [24 x i32] zeroinitializer, align 16
// CHECK: @_unnamed_cfstring_{{.*}} = private global %struct.__NSConstantString_tag { i32* getelementptr inbounds ([24 x i32], [24 x i32]* @__CFConstantStringClassReference, i32 0, i32 0)

// CHECK-LABEL: define internal void @_inlineFunction()
// CHECK:  [[ZERO:%.*]] = load %struct._class_t*, %struct._class_t** @"OBJC_CLASSLIST_REFERENCES_
// CHECK-NEXT:   [[ONE:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_
// CHECK-NEXT:   [[TWO:%.*]] = bitcast %struct._class_t* [[ZERO]] to i8*
// CHECK-NEXT:   call void (i8*, i8*, [[T:%.*]]*, ...) bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, [[T:%.*]]*, ...)*)(i8* [[TWO]], i8* [[ONE]], [[T:%.*]]* bitcast (%struct.__NSConstantString_tag* @_unnamed_cfstring_{{.*}} to [[T:%.*]]*))
// CHECK-NEXT:   ret void
