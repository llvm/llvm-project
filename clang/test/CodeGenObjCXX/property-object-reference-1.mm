// RUN: %clang_cc1 -no-opaque-pointers -x objective-c++ %s -triple x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s
// rdar://6137845

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

// CHECK: [[cppObjectaddr:%.*]] = alloca %struct.TCPPObject*, align 8
// CHECK: store %struct.TCPPObject* [[cppObject:%.*]], %struct.TCPPObject** [[cppObjectaddr]], align 8
// CHECK:  [[THREE:%.*]] = load %struct.TCPPObject*, %struct.TCPPObject** [[cppObjectaddr]], align 8
// CHECK:  [[FOUR:%.*]] = bitcast %struct.TCPPObject* [[THREE]] to i8*
// CHECK:  call void @objc_copyStruct(i8* noundef [[TWO:%.*]], i8* noundef [[FOUR]], i64 noundef 256, i1 noundef zeroext true, i1 noundef zeroext false)
