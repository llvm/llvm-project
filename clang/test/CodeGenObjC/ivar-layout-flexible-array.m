// RUN: %clang_cc1 -triple x86_64-apple-darwin -Wno-objc-root-class -fobjc-arc -emit-llvm -o - %s | FileCheck %s

@interface FlexibleArrayMember {
  char flexible_array[][4][2];
}
@end
@implementation FlexibleArrayMember
@end
// CHECK: @OBJC_METH_VAR_NAME_{{.*}} = private unnamed_addr constant {{.*}} c"flexible_array\00"
// CHECK-NEXT: @OBJC_METH_VAR_TYPE_{{.*}} = private unnamed_addr constant {{.*}} c"^[4[2c]]\00"


typedef char FlexibleArray[];

struct Packet {
  int size;
  FlexibleArray data;
};

@interface VariableSizeIvar {
  struct Packet flexible_struct;
}
@end
@implementation VariableSizeIvar
@end
// CHECK: @OBJC_METH_VAR_NAME_{{.*}} = private unnamed_addr constant {{.*}} c"flexible_struct\00"
// CHECK-NEXT: @OBJC_METH_VAR_TYPE_{{.*}} = private unnamed_addr constant {{.*}} c"{Packet=\22size\22i\22data\22[0c]}\00"
