// RUN: %clang_cc1 -fblocks -fobjc-gc -triple x86_64-apple-darwin -fobjc-runtime=macosx-fragile-10.5 -emit-llvm %s -o - | \
// RUN: FileCheck %s
// RUN: %clang_cc1 -fblocks -fobjc-gc -triple i386-apple-darwin -fobjc-runtime=macosx-fragile-10.5 -emit-llvm %s -o - | \
// RUN: FileCheck %s

@interface NSObject 
- copy;
@end

int main(void) {
    NSObject *object = 0;
    __weak __block NSObject* weak_object = object;
    void (^callback) (void) = [^{
        if (weak_object)
                [weak_object copy];
    } copy];
    callback();
    return 0;
}

// CHECK: call i8* @objc_read_weak
// CHECK: call i8* @objc_read_weak
