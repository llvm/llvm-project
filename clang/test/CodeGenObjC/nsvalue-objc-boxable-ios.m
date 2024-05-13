// RUN: %clang_cc1 -I %S/Inputs -triple armv7-apple-ios8.0.0 -emit-llvm -O2 -disable-llvm-passes -o - %s | FileCheck %s

#import "nsvalue-boxed-expressions-support.h"

// CHECK:      [[CLASS:@.*]]        = external global %struct._class_t
// CHECK:      [[NSVALUE:@.*]]      = {{.*}}[[CLASS]]{{.*}}
// CHECK:      [[RANGE_STR:.*]]     = {{.*}}_NSRange=II{{.*}}
// CHECK:      [[METH:@.*]]         = private unnamed_addr constant {{.*}}valueWithBytes:objCType:{{.*}}
// CHECK:      [[VALUE_SEL:@.*]]    = {{.*}}[[METH]]{{.*}}
// CHECK:      [[POINT_STR:.*]]     = {{.*}}CGPoint=dd{{.*}}
// CHECK:      [[SIZE_STR:.*]]      = {{.*}}CGSize=dd{{.*}}
// CHECK:      [[RECT_STR:.*]]      = {{.*}}CGRect={CGPoint=dd}{CGSize=dd}}{{.*}}
// CHECK:      [[EDGE_STR:.*]]      = {{.*}}NSEdgeInsets=dddd{{.*}}

// CHECK-LABEL: define{{.*}} void @doRange()
void doRange(void) {
  // CHECK:      [[LOCAL_VAR:%.*]]  = alloca %struct._NSRange{{.*}}
  // CHECK:      [[TEMP_VAR:%.*]]   = alloca %struct._NSRange{{.*}}
  // CHECK:      [[RECV_PTR:%.*]]   = load {{.*}} [[NSVALUE]]
  // CHECK:      call void @llvm.memcpy{{.*}} [[TEMP_VAR]]{{.*}} [[LOCAL_VAR]]{{.*}}
  // CHECK:      [[SEL:%.*]]        = load ptr, ptr [[VALUE_SEL]]
  NSRange ns_range = { .location = 0, .length = 42 };
  // CHECK:      call {{.*objc_msgSend.*}}(ptr noundef [[RECV_PTR]], ptr noundef [[SEL]], ptr noundef [[TEMP_VAR]], ptr noundef {{.*}}[[RANGE_STR]]{{.*}})
  NSValue *range = @(ns_range);
  // CHECK:      ret void
}

// CHECK-LABEL: define{{.*}} void @doPoint()
void doPoint(void) {
  // CHECK:      [[LOCAL_VAR:%.*]]  = alloca %struct.CGPoint{{.*}}
  // CHECK:      [[TEMP_VAR:%.*]]   = alloca %struct.CGPoint{{.*}}
  // CHECK:      [[RECV_PTR:%.*]]   = load {{.*}} [[NSVALUE]]
  // CHECK:      call void @llvm.memcpy{{.*}} [[TEMP_VAR]]{{.*}} [[LOCAL_VAR]]{{.*}}
  // CHECK:      [[SEL:%.*]]        = load ptr, ptr [[VALUE_SEL]]
  CGPoint cg_point = { .x = 42, .y = 24 };
  // CHECK:      call {{.*objc_msgSend.*}}(ptr noundef [[RECV_PTR]], ptr noundef [[SEL]], ptr noundef [[TEMP_VAR]], ptr noundef {{.*}}[[POINT_STR]]{{.*}})
  NSValue *point = @(cg_point);
  // CHECK:      ret void
}

// CHECK-LABEL: define{{.*}} void @doSize()
void doSize(void) {
  // CHECK:      [[LOCAL_VAR:%.*]]  = alloca %struct.CGSize{{.*}}
  // CHECK:      [[TEMP_VAR:%.*]]   = alloca %struct.CGSize{{.*}}
  // CHECK:      [[RECV_PTR:%.*]]   = load {{.*}} [[NSVALUE]]
  // CHECK:      call void @llvm.memcpy{{.*}} [[TEMP_VAR]]{{.*}} [[LOCAL_VAR]]{{.*}}
  // CHECK:      [[SEL:%.*]]        = load ptr, ptr [[VALUE_SEL]]
  CGSize cg_size = { .width = 42, .height = 24 };
  // CHECK:      call {{.*objc_msgSend.*}}(ptr noundef [[RECV_PTR]], ptr noundef [[SEL]], ptr noundef [[TEMP_VAR]], ptr noundef {{.*}}[[SIZE_STR]]{{.*}})
  NSValue *size = @(cg_size);
  // CHECK:      ret void
}

// CHECK-LABEL: define{{.*}} void @doRect()
void doRect(void) {
  // CHECK:      [[LOCAL_VAR:%.*]]  = alloca %struct.CGRect{{.*}}
  // CHECK:      [[TEMP_VAR:%.*]]   = alloca %struct.CGRect{{.*}}
  // CHECK:      [[RECV_PTR:%.*]]   = load {{.*}} [[NSVALUE]]
  // CHECK:      call void @llvm.memcpy{{.*}} [[TEMP_VAR]]{{.*}} [[LOCAL_VAR]]{{.*}}
  // CHECK:      [[SEL:%.*]]        = load ptr, ptr [[VALUE_SEL]]
  CGPoint cg_point = { .x = 42, .y = 24 };
  CGSize cg_size = { .width = 42, .height = 24 };
  CGRect cg_rect = { .origin = cg_point, .size = cg_size };
  // CHECK:      call {{.*objc_msgSend.*}}(ptr noundef [[RECV_PTR]], ptr noundef [[SEL]], ptr noundef [[TEMP_VAR]], ptr{{.*}}[[RECT_STR]]{{.*}})
  NSValue *rect = @(cg_rect);
  // CHECK:      ret void
}

// CHECK-LABEL: define{{.*}} void @doNSEdgeInsets()
void doNSEdgeInsets(void) {
  // CHECK:      [[LOCAL_VAR:%.*]]  = alloca %struct.NSEdgeInsets{{.*}}
  // CHECK:      [[TEMP_VAR:%.*]]   = alloca %struct.NSEdgeInsets{{.*}}
  // CHECK:      [[RECV_PTR:%.*]]   = load {{.*}} [[NSVALUE]]
  // CHECK:      call void @llvm.memcpy{{.*}} [[TEMP_VAR]]{{.*}} [[LOCAL_VAR]]{{.*}}
  // CHECK:      [[SEL:%.*]]        = load ptr, ptr [[VALUE_SEL]]
  NSEdgeInsets ns_edge_insets;
  // CHECK:      call {{.*objc_msgSend.*}}(ptr noundef [[RECV_PTR]], ptr noundef [[SEL]], ptr noundef [[TEMP_VAR]], ptr{{.*}}[[EDGE_STR]]{{.*}})
  NSValue *edge_insets = @(ns_edge_insets);
  // CHECK:      ret void
}

// CHECK-LABEL: define{{.*}} void @doRangeRValue() 
void doRangeRValue(void) {
  // CHECK:     [[COERCE:%.*]]          = alloca %struct._NSRange{{.*}}
  // CHECK:     [[RECV_PTR:%.*]]        = load {{.*}} [[NSVALUE]]
  // CHECK:     call {{.*}} @getRange({{.*}} [[COERCE]])
  // CHECK:     [[SEL:%.*]]             = load ptr, ptr [[VALUE_SEL]]
  // CHECK:     call {{.*objc_msgSend.*}}(ptr noundef [[RECV_PTR]], ptr noundef [[SEL]], ptr noundef [[COERCE]], ptr noundef {{.*}}[[RANGE_STR]]{{.*}})
  NSValue *range_rvalue = @(getRange());
  // CHECK: ret void
}

