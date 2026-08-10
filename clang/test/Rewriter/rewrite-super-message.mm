// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: %clang_cc1 -Wno-address-of-temporary -DKEEP_ATTRIBUTES -D"id=struct objc_object *" -D"Class=struct objc_class *" -D"SEL=void*" -D"__declspec(X)=" -emit-llvm -o - %t-rw.cpp | FileCheck %t-rw.cpp

void *sel_registerName(const char *);

@interface __NSCFType
@end

@interface __NSCFString : __NSCFType
- (const char *)UTF8String;
@end

@implementation __NSCFString
- (const char *)UTF8String {
    return (const char *)[super UTF8String];
}
@end

// CHECK: call ptr @class_getSuperclass

@class NSZone;

@interface NSObject {
}

+ (id)allocWithZone:(NSZone *)zone;
@end


@interface NSArray : NSObject
@end

@implementation NSArray
+ (id)allocWithZone:(NSZone *)zone {
    return [super allocWithZone:zone];
}
@end

@interface XNSArray
{
  Class isa;
}
@end

@class XNSArray;

@interface __NSArray0 : XNSArray
@end

@implementation __NSArray0 @end
