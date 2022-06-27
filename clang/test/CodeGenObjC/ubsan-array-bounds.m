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

@interface FlexibleArray2 {
@public
  char chars[0];
}
@end
@implementation FlexibleArray2 {
@public
  char chars2[0];
}
@end

// CHECK-LABEL: test_FlexibleArray2_1
char test_FlexibleArray2_1(FlexibleArray2 *FA2) {
  // CHECK: !nosanitize
  return FA2->chars[1];
  // CHECK: }
}

// CHECK-LABEL: test_FlexibleArray2_2
char test_FlexibleArray2_2(FlexibleArray2 *FA2) {
  // CHECK-NOT: !nosanitize
  return FA2->chars2[1];
  // CHECK: }
}

@interface FlexibleArray3 {
@public
  char chars[0];
}
@end
@implementation FlexibleArray3 {
@public
  int i;
}
@end

// CHECK-LABEL: test_FlexibleArray3
char test_FlexibleArray3(FlexibleArray3 *FA3) {
  // CHECK: !nosanitize
  return FA3->chars[1];
  // CHECK: }
}
