// RUN: %clang_cc1 -x objective-c -emit-llvm -triple x86_64-apple-macosx10.10.0 -Wno-objc-root-class -fsanitize=array-bounds %s -o - | FileCheck %s

@interface FlexibleArray1 {
@public
  char chars[0];
}
@end
@implementation FlexibleArray1
@end

// CHECK-LABEL: test_FlexibleArray1
char test_FlexibleArray1(FlexibleArray1 *FA1) {
  // CHECK-NOT: !nosanitize
  return FA1->chars[1];
  // CHECK: }
}
