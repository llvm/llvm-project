// RUN: %clang_cc1 -fblocks -triple arm64-apple-darwin %s -emit-llvm -o - | FileCheck %s

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
    // CHECK: call void @objc_msgSend(ptr dead_on_unwind writable sret(%struct.stret) align 4 [[T0:%[^,]+]]
    // CHECK: call void @llvm.memset.p0.i64(ptr align 4 [[T0]], i8 0, i64 400, i1 false)

    s = [Test method];
    // CHECK: call void @objc_msgSend(ptr dead_on_unwind writable sret(%struct.stret) align 4 [[T1:%[^,]+]]
    // CHECK-NOT: call void @llvm.memset.p0.i64(

    [(id)(argc&~255) method];
    // CHECK: call void @objc_msgSend(ptr dead_on_unwind writable sret(%struct.stret) align 4 [[T1:%[^,]+]]
    // CHECK-NOT: call void @llvm.memset.p0.i64(

    [Test method];
    // CHECK: call void @objc_msgSend(ptr dead_on_unwind writable sret(%struct.stret) align 4 [[T1:%[^,]+]]
    // CHECK-NOT: call void @llvm.memset.p0.i64(
}
