// RUN: %clang_cc1 -triple x86_64-apple-macosx11.0.0 \
// RUN:   -fobjc-runtime=macosx-11.0.0 -fobjc-constant-literals \
// RUN:   -fconstant-nsnumber-literals -fconstant-nsarray-literals \
// RUN:   -fconstant-nsdictionary-literals \
// RUN:   -fconstant-integer-number-class=MyIntegerNumber \
// RUN:   -fconstant-float-number-class=MyFloatNumber \
// RUN:   -fconstant-double-number-class=MyDoubleNumber \
// RUN:   -fconstant-array-class=MyArray \
// RUN:   -fconstant-dictionary-class=MyDictionary \
// RUN:   -I %S/Inputs -emit-llvm -o - %s | FileCheck %s

#if __has_feature(objc_constant_literals)

#include "constant-literal-support.h"

@interface MyIntegerNumber : NSConstantIntegerNumber
@end
@interface MyFloatNumber : NSConstantFloatNumber
@end
@interface MyDoubleNumber : NSConstantDoubleNumber
@end
@interface MyArray : NSConstantArray
@end
@interface MyDictionary : NSConstantDictionary
@end

// CHECK: @"OBJC_CLASS_$_MyIntegerNumber" = external global %struct._class_t
// CHECK: @"OBJC_CLASS_$_MyFloatNumber" = external global %struct._class_t
// CHECK: @"OBJC_CLASS_$_MyDoubleNumber" = external global %struct._class_t
// CHECK: @"OBJC_CLASS_$_MyArray" = external global %struct._class_t
// CHECK: @"OBJC_CLASS_$_MyDictionary" = external global %struct._class_t

int main() {
  NSNumber *i = @42;
  NSNumber *f = @1.5f;
  NSNumber *d = @3.14;
  NSArray *a = @[ @"hello" ];
  NSDictionary *dict = @{ @"key" : @"value" };
  return 0;
}

#endif
