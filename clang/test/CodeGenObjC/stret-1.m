// RUN: %clang_cc1 -fblocks -triple arm64-apple-darwin %s -emit-llvm -o - | FileCheck %s
// rdar://12416433

struct stret { int x[100]; };
struct stret one = {{1}};

@interface Test  @end

@implementation Test
+(struct stret) method { return one; }
@end

int main(int argc, const char **argv)
{
    struct stret s;
    s = [(id)(argc&~255) method];
    // CHECK: call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (%struct.stret*, i8*, i8*)*)(%struct.stret* sret(%struct.stret) align 4 [[T0:%[^,]+]]
    // CHECK: [[T0P:%.*]] = bitcast %struct.stret* [[T0]] to i8*
    // CHECK: call void @llvm.memset.p0i8.i64(i8* align 4 [[T0P]], i8 0, i64 400, i1 false)

    s = [Test method];
    // CHECK: call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (%struct.stret*, i8*, i8*)*)(%struct.stret* sret(%struct.stret) align 4 [[T1:%[^,]+]]
    // CHECK-NOT: call void @llvm.memset.p0i8.i64(

    [(id)(argc&~255) method];
    // CHECK: call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (%struct.stret*, i8*, i8*)*)(%struct.stret* sret(%struct.stret) align 4 [[T1:%[^,]+]]
    // CHECK-NOT: call void @llvm.memset.p0i8.i64(

    [Test method];
    // CHECK: call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (%struct.stret*, i8*, i8*)*)(%struct.stret* sret(%struct.stret) align 4 [[T1:%[^,]+]]
    // CHECK-NOT: call void @llvm.memset.p0i8.i64(
}
