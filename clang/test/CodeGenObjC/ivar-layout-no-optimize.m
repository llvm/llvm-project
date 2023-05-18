// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -fobjc-gc -triple x86_64-apple-darwin -fobjc-runtime=macosx-fragile-10.5 -S %s -o %t-64.s
// RUN: FileCheck -check-prefix CHECK-LP64 --input-file=%t-64.s %s
// RUN: %clang_cc1 -x objective-c++ -fobjc-gc -triple x86_64-apple-darwin -fobjc-runtime=macosx-fragile-10.5 -S %s -o %t-64.s
// RUN: FileCheck -check-prefix CHECK-LP64 --input-file=%t-64.s %s

@interface NSObject {
  id isa;
}
@end

@interface AllPointers : NSObject {
    id foo;
    void *__strong bar;    NSObject *bletch;}
@end
@implementation AllPointers
@end

// CHECK-LP64: L_OBJC_CLASS_NAME_.6:
// CHECK-LP64-NEXT: .asciz	"\004"
