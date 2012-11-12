// REQUIRES: x86-64-registered-target
// RUN: %clang_cc1 -fobjc-gc -triple x86_64-apple-darwin -fobjc-runtime=macosx-fragile-10.5 -O0 -S %s -o %t-64.s
// RUN: FileCheck -check-prefix LP64 --input-file=%t-64.s %s

// rdar://8800513
@interface NSObject {
  id isa;
}
@end

typedef struct {
    id b;
} st;

@interface Test : NSObject {
    int a;
    st b[0];
}
@end

@implementation Test @end
// CHECK-LP64: L_OBJC_CLASS_NAME_4:
// CHECK-LP64-NEXT: .asciz      "\001\020"
