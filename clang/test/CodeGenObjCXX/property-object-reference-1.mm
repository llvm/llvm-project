// RUN: %clang_cc1 -x objective-c++ %s -triple x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

struct TCPPObject
{
 TCPPObject(const TCPPObject& inObj);
 TCPPObject();
 ~TCPPObject();
 int filler[64];
};


@interface MyDocument 
{
@private
 TCPPObject _cppObject;
}
@property (atomic, assign, readwrite) const TCPPObject& cppObject;
@end

@implementation MyDocument

@synthesize cppObject = _cppObject;

@end

// CHECK: [[cppObjectaddr:%cppObject.addr]] = alloca ptr, align 8
// CHECK: store ptr [[cppObject:%.*]], ptr [[cppObjectaddr]], align 8
// CHECK:  [[THREE:%.*]] = load ptr, ptr [[cppObjectaddr]], align 8
// CHECK:  call void @objc_copyStruct(ptr noundef [[TWO:%.*]], ptr noundef [[THREE]], i64 noundef 256, i1 noundef zeroext true, i1 noundef zeroext false)
