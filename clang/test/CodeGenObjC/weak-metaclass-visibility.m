// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple armv7-apple-darwin10 -emit-llvm  -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios -emit-llvm -o - %s | FileCheck %s
// rdar://16206443

@interface NSObject 
- (void) finalize;
+ (void) class;
@end

__attribute__((availability(macosx,introduced=9876.5)))
@interface MyClass : NSObject
+ (void)someClassMethod;
- (void)someInstanceMethod;
@end

@implementation MyClass
+ (void)someClassMethod {
}

- (void)someInstanceMethod {
    [MyClass someClassMethod];
    [super finalize];
}
@end

void kit(void)
{
    MyClass *wrapper = [MyClass alloc];
}

// CHECK: @"OBJC_CLASS_$_MyClass" ={{.*}} global %struct._class_t
// CHECK: @"OBJC_METACLASS_$_NSObject" = external global %struct._class_t
// CHECK: @"OBJC_METACLASS_$_MyClass" ={{.*}} global %struct._class_t
// CHECK: @"OBJC_CLASS_$_NSObject" = external global %struct._class_t

// rdar://16529125
__attribute__((weak_import))
@interface NSURLQueryItem : NSObject
@end

@implementation NSURLQueryItem (hax)
+(void)classmethod { [super class]; }
@end

// CHECK: @"OBJC_METACLASS_$_NSURLQueryItem" = extern_weak global
// CHECK: @"OBJC_CLASS_$_NSURLQueryItem" = extern_weak global

// rdar://17633301
__attribute__((visibility("default"))) __attribute__((availability(ios,introduced=9876.5)))
@interface AVScheduledAudioParameters @end

@interface XXXX : AVScheduledAudioParameters
@end

@implementation AVScheduledAudioParameters @end
@implementation XXXX @end

// CHECK: @"OBJC_CLASS_$_AVScheduledAudioParameters" ={{.*}} global %struct._class_t
// CHECK: @"OBJC_METACLASS_$_AVScheduledAudioParameters" ={{.*}} global %struct._class_t 
