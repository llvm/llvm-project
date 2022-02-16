// RUN: %clang_cc1 -disable-noundef-analysis -triple x86_64-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -disable-noundef-analysis -x objective-c++ -triple x86_64-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o - %s | FileCheck %s

// rdar: // 8399655
@interface TestClass
@property (readonly) int myProperty;
- (int)myProperty;
- (double)myGetter;
@end

void FUNC (void) {
    TestClass *obj;
    (void)obj.myProperty; 
    (void)obj.myGetter; 
}

// CHECK: call i32 bitcast
// CHECK: call double bitcast
