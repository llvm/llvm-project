// RUN: %clang_cc1 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK-MAC
// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK-MAC-NF
// RUN: %clang_cc1 -fobjc-runtime=gcc -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK-GNU
// RUN: %clang_cc1 -fobjc-runtime=gnustep -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK-GNU-NF
// RUN: %clang_cc1 -fobjc-runtime=gnustep-2.2 -fobjc-dispatch-method=non-legacy -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK-MAC

typedef struct {
  int x;
  int y;
  int z[10];
} MyPoint;

void f0(id a) {
  int i;
  MyPoint pt = { 1, 2};

  // CHECK-MAC: call {{.*}} @objc_msgSend
  // CHECK-MAC-NF: call {{.*}} @objc_msgSend
  // CHECK-GNU: call {{.*}} @objc_msg_lookup(
  // CHECK-GNU-NF: call {{.*}} @objc_msg_lookup_sender(
  [a print0];

  // CHECK-MAC: call {{.*}} @objc_msgSend
  // CHECK-MAC-NF: call {{.*}} @objc_msgSend
  // CHECK-GNU: call {{.*}} @objc_msg_lookup(
  // CHECK-GNU-NF: call {{.*}} @objc_msg_lookup_sender(
  [a print1: 10];

  // CHECK-MAC: call {{.*}} @objc_msgSend
  // CHECK-MAC-NF: call {{.*}} @objc_msgSend
  // CHECK-GNU: call {{.*}} @objc_msg_lookup(
  // CHECK-GNU-NF: call {{.*}} @objc_msg_lookup_sender(
  [a print2: 10 and: "hello" and: 2.2];

  // CHECK-MAC: call {{.*}} @objc_msgSend
  // CHECK-MAC-NF: call {{.*}} @objc_msgSend
  // CHECK-GNU: call {{.*}} @objc_msg_lookup(
  // CHECK-GNU-NF: call {{.*}} @objc_msg_lookup_sender(
  [a takeStruct: pt ];
  
  void *s = @selector(print0);
  for (i=0; i<2; ++i)
    // CHECK-MAC: call {{.*}} @objc_msgSend
    // CHECK-MAC-NF: call {{.*}} @objc_msgSend
    // CHECK-GNU: call {{.*}} @objc_msg_lookup(
    // CHECK-GNU-NF: call {{.*}} @objc_msg_lookup_sender(
    [a performSelector:s];
}
