// Test without serialization:
// RUN: %clang_cc1 -triple x86_64-pc-linux -ast-dump %s \
// RUN: | FileCheck --strict-whitespace %s

// Test with serialization:
// RUN: %clang_cc1 -triple x86_64-pc-linux -emit-pch -o %t %s
// RUN: %clang_cc1 -x objective-c -triple x86_64-pc-linux -include-pch %t -ast-dump-all /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck --strict-whitespace %s


@interface Adder
- (float) sum: (float)x with: (float)y __attribute((optnone));
@end

#pragma float_control(precise, off)

@implementation Adder
- (float) sum: (float)x with: (float)y __attribute((optnone)) {
  return x + y;
}

@end

// CHECK-LABEL: ObjCImplementationDecl {{.*}} Adder
// CHECK:         ObjCMethodDecl {{.*}} - sum:with: 'float'
// CHECK:           CompoundStmt {{.*}} FPContractMode=1 MathErrno=1
// CHECK-NEXT:        ReturnStmt
// CHECK-NEXT:          BinaryOperator {{.*}} 'float' '+' FPContractMode=1 MathErrno=1
