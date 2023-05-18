// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s

@interface NSObject @end

@class NSString;

int __CFConstantStringClassReference[24];

@interface Bar : NSObject
+(void)format:(NSString *)format,...;
@end

@interface Foo : NSObject
@end


static inline void _inlineFunction(void) {
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
// CHECK: @_unnamed_cfstring_{{.*}} = private global %struct.__NSConstantString_tag { ptr @__CFConstantStringClassReference

// CHECK-LABEL: define internal void @_inlineFunction()
// CHECK:  [[ZERO:%.*]] = load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_
// CHECK-NEXT:   [[SEL:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_
// CHECK-NEXT:   call void (ptr, ptr, ptr, ...) @objc_msgSend(ptr noundef [[ZERO]], ptr noundef [[SEL]], ptr noundef @_unnamed_cfstring_{{.*}})
// CHECK-NEXT:   ret void
