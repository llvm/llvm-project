// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5  -emit-llvm -o - %s | FileCheck %s
// PR7390

// CHECK: @[[setprioname:[^ ]*]] = {{.*}}"setPriority:
// CHECK-NEXT: @[[setpriosel:[^ ]*]] = {{.*}}[[setprioname]]
@interface NSObject
- (void)respondsToSelector:(const SEL &)s ps:(SEL *)s1;
- (void)setPriority:(int)p;
- (void)Meth;
@end

@implementation NSObject

// CHECK-LABEL: define internal void @"\01-[NSObject Meth]"(
- (void)Meth {
// CHECK: call void @objc_msgSend{{.*}}, ptr noundef @[[setpriosel]])
  [self respondsToSelector:@selector(setPriority:) ps:&@selector(setPriority:)];
}
- (void)setPriority:(int)p {
}
- (void)respondsToSelector:(const SEL &)s ps:(SEL *)s1 {
}
@end
