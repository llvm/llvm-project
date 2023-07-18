// RUN: %clang_cc1 -fobjc-gc -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o - %s | FileCheck %s
// rdar://8681766

@interface NSArray 
- (NSArray*) retain;
- (void) release;
@end

void NSAssignArray(NSArray*& target, NSArray* newValue)
{
        if (target == newValue)
                return;

        NSArray* oldValue = target;

        target = [newValue retain];

        [oldValue release];
}
// CHECK: {{call.* @objc_assign_strongCast}}
