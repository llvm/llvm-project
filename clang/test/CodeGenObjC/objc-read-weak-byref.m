// REQUIRES: x86-registered-target,x86-64-registered-target
// RUN: %clang_cc1 -fblocks -fobjc-gc -triple x86_64-apple-darwin -fobjc-runtime=macosx-fragile-10.5 -S %s -o %t-64.s
// RUN: FileCheck -check-prefix LP64 --input-file=%t-64.s %s
// RUN: %clang_cc1 -fblocks -fobjc-gc -triple i386-apple-darwin -fobjc-runtime=macosx-fragile-10.5 -S %s -o %t-32.s
// RUN: FileCheck -check-prefix LP32 --input-file=%t-32.s %s

@interface NSObject 
- copy;
@end

int main() {
    NSObject *object = 0;
    __weak __block NSObject* weak_object = object;
    void (^callback) (void) = [^{
        if (weak_object)
                [weak_object copy];
    } copy];
    callback();
    return 0;
}

// CHECK-LP64: callq    _objc_read_weak
// CHECK-LP64: callq    _objc_read_weak

// CHECK-LP32: calll     L_objc_read_weak
// CHECK-LP32: calll     L_objc_read_weak
