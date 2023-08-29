// RUN: %clang_cc1 -E %s -o %t.mm
// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %t.mm -o - | FileCheck %s 

@class NSString;

@interface NSObject {
    Class isa;
}
@end

@interface Sub : NSObject {
    int subIvar;
    NSString *nsstring;
@private
    id PrivateIvar;
}
@end

@implementation Sub 
- (id) MyNSString { return subIvar ? PrivateIvar : nsstring; }
@end

@interface NSString @end
@implementation NSString @end

// CHECK: __declspec(allocate(".objc_ivar$B")) extern "C" __declspec(dllimport) unsigned long OBJC_IVAR_$_Sub$subIvar;
// CHECK: __declspec(allocate(".objc_ivar$B")) extern "C" unsigned long OBJC_IVAR_$_Sub$PrivateIvar;
// CHECK: __declspec(allocate(".objc_ivar$B")) extern "C" __declspec(dllimport) unsigned long OBJC_IVAR_$_Sub$nsstring;
// CHECK: #pragma warning(disable:4273)
// CHECK: __declspec(allocate(".objc_ivar$B")) extern "C" __declspec(dllexport) unsigned long int OBJC_IVAR_$_Sub$subIvar
// CHECK: __declspec(allocate(".objc_ivar$B")) extern "C" __declspec(dllexport) unsigned long int OBJC_IVAR_$_Sub$nsstring
// CHECK: __declspec(allocate(".objc_ivar$B")) extern "C" unsigned long int OBJC_IVAR_$_Sub$PrivateIvar
// CHECK: extern "C" __declspec(dllimport) struct _class_t OBJC_METACLASS_$_NSObject;
// CHECK: extern "C" __declspec(dllexport) struct _class_t OBJC_METACLASS_$_Sub
// CHECK: extern "C" __declspec(dllimport) struct _class_t OBJC_CLASS_$_NSObject;
// CHECK: extern "C" __declspec(dllexport) struct _class_t OBJC_CLASS_$_Sub
// CHECK: extern "C" __declspec(dllexport) struct _class_t OBJC_CLASS_$_NSString;
// CHECK: extern "C" __declspec(dllexport) struct _class_t OBJC_METACLASS_$_NSString
// CHECK: extern "C" __declspec(dllexport) struct _class_t OBJC_CLASS_$_NSString
