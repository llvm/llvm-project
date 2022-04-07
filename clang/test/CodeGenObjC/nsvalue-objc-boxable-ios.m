// RUN: %clang_cc1 -no-opaque-pointers -I %S/Inputs -triple armv7-apple-ios8.0.0 -emit-llvm -O2 -disable-llvm-passes -o - %s | FileCheck %s

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
  // CHECK:      [[TEMP_CAST:%.*]]  = bitcast %struct._NSRange* [[TEMP_VAR]]{{.*}}
  // CHECK:      [[LOCAL_CAST:%.*]] = bitcast %struct._NSRange* [[LOCAL_VAR]]{{.*}}
  // CHECK:      call void @llvm.memcpy{{.*}} [[TEMP_CAST]]{{.*}} [[LOCAL_CAST]]{{.*}}
  // CHECK:      [[PARAM_CAST:%.*]] = bitcast %struct._NSRange* [[TEMP_VAR]]{{.*}}
  // CHECK:      [[RECV:%.*]]       = bitcast %struct._class_t* [[RECV_PTR]] to i8*
  // CHECK:      [[SEL:%.*]]        = load i8*, i8** [[VALUE_SEL]]
  NSRange ns_range = { .location = 0, .length = 42 };
  // CHECK:      call {{.*objc_msgSend.*}}(i8* noundef [[RECV]], i8* noundef [[SEL]], i8* noundef [[PARAM_CAST]], i8* noundef {{.*}}[[RANGE_STR]]{{.*}})
  NSValue *range = @(ns_range);
  // CHECK:      ret void
}

// CHECK-LABEL: define{{.*}} void @doPoint()
void doPoint(void) {
  // CHECK:      [[LOCAL_VAR:%.*]]  = alloca %struct.CGPoint{{.*}}
  // CHECK:      [[TEMP_VAR:%.*]]   = alloca %struct.CGPoint{{.*}}
  // CHECK:      [[RECV_PTR:%.*]]   = load {{.*}} [[NSVALUE]]
  // CHECK:      [[TEMP_CAST:%.*]]  = bitcast %struct.CGPoint* [[TEMP_VAR]]{{.*}}
  // CHECK:      [[LOCAL_CAST:%.*]] = bitcast %struct.CGPoint* [[LOCAL_VAR]]{{.*}}
  // CHECK:      call void @llvm.memcpy{{.*}} [[TEMP_CAST]]{{.*}} [[LOCAL_CAST]]{{.*}}
  // CHECK:      [[PARAM_CAST:%.*]] = bitcast %struct.CGPoint* [[TEMP_VAR]]{{.*}}
  // CHECK:      [[RECV:%.*]]       = bitcast %struct._class_t* [[RECV_PTR]] to i8*
  // CHECK:      [[SEL:%.*]]        = load i8*, i8** [[VALUE_SEL]]
  CGPoint cg_point = { .x = 42, .y = 24 };
  // CHECK:      call {{.*objc_msgSend.*}}(i8* noundef [[RECV]], i8* noundef [[SEL]], i8* noundef [[PARAM_CAST]], i8* noundef {{.*}}[[POINT_STR]]{{.*}})
  NSValue *point = @(cg_point);
  // CHECK:      ret void
}

// CHECK-LABEL: define{{.*}} void @doSize()
void doSize(void) {
  // CHECK:      [[LOCAL_VAR:%.*]]  = alloca %struct.CGSize{{.*}}
  // CHECK:      [[TEMP_VAR:%.*]]   = alloca %struct.CGSize{{.*}}
  // CHECK:      [[RECV_PTR:%.*]]   = load {{.*}} [[NSVALUE]]
  // CHECK:      [[TEMP_CAST:%.*]]  = bitcast %struct.CGSize* [[TEMP_VAR]]{{.*}}
  // CHECK:      [[LOCAL_CAST:%.*]] = bitcast %struct.CGSize* [[LOCAL_VAR]]{{.*}}
  // CHECK:      call void @llvm.memcpy{{.*}} [[TEMP_CAST]]{{.*}} [[LOCAL_CAST]]{{.*}}
  // CHECK:      [[PARAM_CAST:%.*]] = bitcast %struct.CGSize* [[TEMP_VAR]]{{.*}}
  // CHECK:      [[RECV:%.*]]       = bitcast %struct._class_t* [[RECV_PTR]] to i8*
  // CHECK:      [[SEL:%.*]]        = load i8*, i8** [[VALUE_SEL]]
  CGSize cg_size = { .width = 42, .height = 24 };
  // CHECK:      call {{.*objc_msgSend.*}}(i8* noundef [[RECV]], i8* noundef [[SEL]], i8* noundef [[PARAM_CAST]], i8* noundef {{.*}}[[SIZE_STR]]{{.*}})
  NSValue *size = @(cg_size);
  // CHECK:      ret void
}

// CHECK-LABEL: define{{.*}} void @doRect()
void doRect(void) {
  // CHECK:      [[LOCAL_VAR:%.*]]  = alloca %struct.CGRect{{.*}}
  // CHECK:      [[TEMP_VAR:%.*]]   = alloca %struct.CGRect{{.*}}
  // CHECK:      [[RECV_PTR:%.*]]   = load {{.*}} [[NSVALUE]]
  // CHECK:      [[TEMP_CAST:%.*]]  = bitcast %struct.CGRect* [[TEMP_VAR]]{{.*}}
  // CHECK:      [[LOCAL_CAST:%.*]] = bitcast %struct.CGRect* [[LOCAL_VAR]]{{.*}}
  // CHECK:      call void @llvm.memcpy{{.*}} [[TEMP_CAST]]{{.*}} [[LOCAL_CAST]]{{.*}}
  // CHECK:      [[PARAM_CAST:%.*]] = bitcast %struct.CGRect* [[TEMP_VAR]]{{.*}}
  // CHECK:      [[RECV:%.*]]       = bitcast %struct._class_t* [[RECV_PTR]] to i8*
  // CHECK:      [[SEL:%.*]]        = load i8*, i8** [[VALUE_SEL]]
  CGPoint cg_point = { .x = 42, .y = 24 };
  CGSize cg_size = { .width = 42, .height = 24 };
  CGRect cg_rect = { .origin = cg_point, .size = cg_size };
  // CHECK:      call {{.*objc_msgSend.*}}(i8* noundef [[RECV]], i8* noundef [[SEL]], i8* noundef [[PARAM_CAST]], i8*{{.*}}[[RECT_STR]]{{.*}})
  NSValue *rect = @(cg_rect);
  // CHECK:      ret void
}

// CHECK-LABEL: define{{.*}} void @doNSEdgeInsets()
void doNSEdgeInsets(void) {
  // CHECK:      [[LOCAL_VAR:%.*]]  = alloca %struct.NSEdgeInsets{{.*}}
  // CHECK:      [[TEMP_VAR:%.*]]   = alloca %struct.NSEdgeInsets{{.*}}
  // CHECK:      [[RECV_PTR:%.*]]   = load {{.*}} [[NSVALUE]]
  // CHECK:      [[TEMP_CAST:%.*]]  = bitcast %struct.NSEdgeInsets* [[TEMP_VAR]]{{.*}}
  // CHECK:      [[LOCAL_CAST:%.*]] = bitcast %struct.NSEdgeInsets* [[LOCAL_VAR]]{{.*}}
  // CHECK:      call void @llvm.memcpy{{.*}} [[TEMP_CAST]]{{.*}} [[LOCAL_CAST]]{{.*}}
  // CHECK:      [[PARAM_CAST:%.*]] = bitcast %struct.NSEdgeInsets* [[TEMP_VAR]]{{.*}}
  // CHECK:      [[RECV:%.*]]       = bitcast %struct._class_t* [[RECV_PTR]] to i8*
  // CHECK:      [[SEL:%.*]]        = load i8*, i8** [[VALUE_SEL]]
  NSEdgeInsets ns_edge_insets;
  // CHECK:      call {{.*objc_msgSend.*}}(i8* noundef [[RECV]], i8* noundef [[SEL]], i8* noundef [[PARAM_CAST]], i8*{{.*}}[[EDGE_STR]]{{.*}})
  NSValue *edge_insets = @(ns_edge_insets);
  // CHECK:      ret void
}

// CHECK-LABEL: define{{.*}} void @doRangeRValue() 
void doRangeRValue(void) {
  // CHECK:     [[COERCE:%.*]]          = alloca %struct._NSRange{{.*}}
  // CHECK:     [[RECV_PTR:%.*]]        = load {{.*}} [[NSVALUE]]
  // CHECK:     call {{.*}} @getRange({{.*}} [[COERCE]])
  // CHECK:     [[COERCE_CAST:%.*]]     = bitcast %struct._NSRange* [[COERCE]]{{.*}}
  // CHECK:     [[RECV:%.*]]            = bitcast %struct._class_t* [[RECV_PTR]] to i8*
  // CHECK:     [[SEL:%.*]]             = load i8*, i8** [[VALUE_SEL]]
  // CHECK:     call {{.*objc_msgSend.*}}(i8* noundef [[RECV]], i8* noundef [[SEL]], i8* noundef [[COERCE_CAST]], i8* noundef {{.*}}[[RANGE_STR]]{{.*}})
  NSValue *range_rvalue = @(getRange());
  // CHECK: ret void
}

