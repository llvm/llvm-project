// RUN: %clang_cc1 -I %S/Inputs -triple x86_64-apple-macosx -emit-llvm -fobjc-arc -O2 -disable-llvm-passes -o - %s | FileCheck %s

#import "nsvalue-boxed-expressions-support.h"

// CHECK:      [[CLASS:@.*]]        = external global %struct._class_t
// CHECK:      [[NSVALUE:@.*]]      = {{.*}}[[CLASS]]{{.*}}
// CHECK:      [[RANGE_STR:.*]]     = {{.*}}_NSRange=QQ{{.*}}
// CHECK:      [[METH:@.*]]         = private unnamed_addr constant {{.*}}valueWithBytes:objCType:{{.*}}
// CHECK:      [[VALUE_SEL:@.*]]    = {{.*}}[[METH]]{{.*}}
// CHECK:      [[POINT_STR:.*]]     = {{.*}}_NSPoint=dd{{.*}}
// CHECK:      [[SIZE_STR:.*]]      = {{.*}}_NSSize=dd{{.*}}
// CHECK:      [[RECT_STR:.*]]      = {{.*}}_NSRect={_NSPoint=dd}{_NSSize=dd}}{{.*}}
// CHECK:      [[EDGE_STR:.*]]      = {{.*}}NSEdgeInsets=dddd{{.*}}

// CHECK-LABEL: define void @doRange()
void doRange() {
  // CHECK:      [[LOCAL_VAR:%.*]]  = alloca %struct._NSRange{{.*}}
  // CHECK:      [[TEMP_VAR:%.*]]   = alloca %struct._NSRange{{.*}}
  // CHECK:      [[RECV_PTR:%.*]]   = load {{.*}} [[NSVALUE]]
  // CHECK:      [[TEMP_CAST:%.*]]  = bitcast %struct._NSRange* [[TEMP_VAR]]{{.*}}
  // CHECK:      [[LOCAL_CAST:%.*]] = bitcast %struct._NSRange* [[LOCAL_VAR]]{{.*}}
  // CHECK:      call void @llvm.memcpy{{.*}} [[TEMP_CAST]]{{.*}} [[LOCAL_CAST]]{{.*}}
  // CHECK:      [[PARAM_CAST:%.*]] = bitcast %struct._NSRange* [[TEMP_VAR]]{{.*}}
  // CHECK:      [[SEL:%.*]]        = load i8*, i8** [[VALUE_SEL]]
  // CHECK:      [[RECV:%.*]]       = bitcast %struct._class_t* [[RECV_PTR]] to i8*
  NSRange ns_range = { .location = 0, .length = 42 };
  // CHECK:      call {{.*objc_msgSend.*}}(i8* [[RECV]], i8* [[SEL]], i8* [[PARAM_CAST]], i8* {{.*}}[[RANGE_STR]]{{.*}})
  // CHECK:      call i8* @llvm.objc.retainAutoreleasedReturnValue
  NSValue *range = @(ns_range);
  // CHECK:      call void @llvm.objc.release
  // CHECK:      ret void
}

// CHECK-LABEL: define void @doPoint()
void doPoint() {
  // CHECK:      [[LOCAL_VAR:%.*]]  = alloca %struct._NSPoint{{.*}}
  // CHECK:      [[TEMP_VAR:%.*]]   = alloca %struct._NSPoint{{.*}}
  // CHECK:      [[RECV_PTR:%.*]]   = load {{.*}} [[NSVALUE]]
  // CHECK:      [[TEMP_CAST:%.*]]  = bitcast %struct._NSPoint* [[TEMP_VAR]]{{.*}}
  // CHECK:      [[LOCAL_CAST:%.*]] = bitcast %struct._NSPoint* [[LOCAL_VAR]]{{.*}}
  // CHECK:      call void @llvm.memcpy{{.*}} [[TEMP_CAST]]{{.*}} [[LOCAL_CAST]]{{.*}}
  // CHECK:      [[PARAM_CAST:%.*]] = bitcast %struct._NSPoint* [[TEMP_VAR]]{{.*}}
  // CHECK:      [[SEL:%.*]]        = load i8*, i8** [[VALUE_SEL]]
  // CHECK:      [[RECV:%.*]]       = bitcast %struct._class_t* [[RECV_PTR]] to i8*
  NSPoint ns_point = { .x = 42, .y = 24 };
  // CHECK:      call {{.*objc_msgSend.*}}(i8* [[RECV]], i8* [[SEL]], i8* [[PARAM_CAST]], i8* {{.*}}[[POINT_STR]]{{.*}})
  // CHECK:      call i8* @llvm.objc.retainAutoreleasedReturnValue
  NSValue *point = @(ns_point);
  // CHECK:      call void @llvm.objc.release
  // CHECK:      ret void
}

// CHECK-LABEL: define void @doSize()
void doSize() {
  // CHECK:      [[LOCAL_VAR:%.*]]  = alloca %struct._NSSize{{.*}}
  // CHECK:      [[TEMP_VAR:%.*]]   = alloca %struct._NSSize{{.*}}
  // CHECK:      [[RECV_PTR:%.*]]   = load {{.*}} [[NSVALUE]]
  // CHECK:      [[TEMP_CAST:%.*]]  = bitcast %struct._NSSize* [[TEMP_VAR]]{{.*}}
  // CHECK:      [[LOCAL_CAST:%.*]] = bitcast %struct._NSSize* [[LOCAL_VAR]]{{.*}}
  // CHECK:      call void @llvm.memcpy{{.*}} [[TEMP_CAST]]{{.*}} [[LOCAL_CAST]]{{.*}}
  // CHECK:      [[PARAM_CAST:%.*]] = bitcast %struct._NSSize* [[TEMP_VAR]]{{.*}}
  // CHECK:      [[SEL:%.*]]        = load i8*, i8** [[VALUE_SEL]]
  // CHECK:      [[RECV:%.*]]       = bitcast %struct._class_t* [[RECV_PTR]] to i8*
  NSSize ns_size = { .width = 42, .height = 24 };
  // CHECK:      call {{.*objc_msgSend.*}}(i8* [[RECV]], i8* [[SEL]], i8* [[PARAM_CAST]], i8* {{.*}}[[SIZE_STR]]{{.*}})
  // CHECK:      call i8* @llvm.objc.retainAutoreleasedReturnValue
  NSValue *size = @(ns_size);
  // CHECK:      call void @llvm.objc.release
  // CHECK:      ret void
}

// CHECK-LABEL: define void @doRect()
void doRect() {
  // CHECK:      [[LOCAL_VAR:%.*]]  = alloca %struct._NSRect{{.*}}
  // CHECK:      [[TEMP_VAR:%.*]]   = alloca %struct._NSRect{{.*}}
  // CHECK:      [[RECV_PTR:%.*]]   = load {{.*}} [[NSVALUE]]
  // CHECK:      [[TEMP_CAST:%.*]]  = bitcast %struct._NSRect* [[TEMP_VAR]]{{.*}}
  // CHECK:      [[LOCAL_CAST:%.*]] = bitcast %struct._NSRect* [[LOCAL_VAR]]{{.*}}
  // CHECK:      call void @llvm.memcpy{{.*}} [[TEMP_CAST]]{{.*}} [[LOCAL_CAST]]{{.*}}
  // CHECK:      [[PARAM_CAST:%.*]] = bitcast %struct._NSRect* [[TEMP_VAR]]{{.*}}
  // CHECK:      [[SEL:%.*]]        = load i8*, i8** [[VALUE_SEL]]
  // CHECK:      [[RECV:%.*]]       = bitcast %struct._class_t* [[RECV_PTR]] to i8*
  NSPoint ns_point = { .x = 42, .y = 24 };
  NSSize ns_size = { .width = 42, .height = 24 };
  NSRect ns_rect = { .origin = ns_point, .size = ns_size };
  // CHECK:      call {{.*objc_msgSend.*}}(i8* [[RECV]], i8* [[SEL]], i8* [[PARAM_CAST]], i8*{{.*}}[[RECT_STR]]{{.*}})
  // CHECK:      call i8* @llvm.objc.retainAutoreleasedReturnValue
  NSValue *rect = @(ns_rect);
  // CHECK:      call void @llvm.objc.release
  // CHECK:      ret void
}

// CHECK-LABEL: define void @doNSEdgeInsets()
void doNSEdgeInsets() {
  // CHECK:      [[LOCAL_VAR:%.*]]  = alloca %struct.NSEdgeInsets{{.*}}
  // CHECK:      [[TEMP_VAR:%.*]]   = alloca %struct.NSEdgeInsets{{.*}}
  // CHECK:      [[RECV_PTR:%.*]]   = load {{.*}} [[NSVALUE]]
  // CHECK:      [[TEMP_CAST:%.*]]  = bitcast %struct.NSEdgeInsets* [[TEMP_VAR]]{{.*}}
  // CHECK:      [[LOCAL_CAST:%.*]] = bitcast %struct.NSEdgeInsets* [[LOCAL_VAR]]{{.*}}
  // CHECK:      call void @llvm.memcpy{{.*}} [[TEMP_CAST]]{{.*}} [[LOCAL_CAST]]{{.*}}
  // CHECK:      [[PARAM_CAST:%.*]] = bitcast %struct.NSEdgeInsets* [[TEMP_VAR]]{{.*}}
  // CHECK:      [[SEL:%.*]]        = load i8*, i8** [[VALUE_SEL]]
  // CHECK:      [[RECV:%.*]]       = bitcast %struct._class_t* [[RECV_PTR]] to i8*
  NSEdgeInsets ns_edge_insets;
  // CHECK:      call {{.*objc_msgSend.*}}(i8* [[RECV]], i8* [[SEL]], i8* [[PARAM_CAST]], i8*{{.*}}[[EDGE_STR]]{{.*}})
  // CHECK:      call i8* @llvm.objc.retainAutoreleasedReturnValue
  NSValue *edge_insets = @(ns_edge_insets);
  // CHECK:      call void @llvm.objc.release
  // CHECK:      ret void
}

// CHECK-LABEL: define void @doRangeRValue() 
void doRangeRValue() {
  // CHECK:     [[COERCE:%.*]]          = alloca %struct._NSRange{{.*}}
  // CHECK:     [[RECV_PTR:%.*]]        = load {{.*}} [[NSVALUE]]
  // CHECK:     [[RVAL:%.*]]            = call {{.*}} @getRange()
  // CHECK:     [[COERCE_CAST:%.*]]     = bitcast %struct._NSRange* [[COERCE]]{{.*}}
  // CHECK:     [[COERCE_CAST_PTR:%.*]] = getelementptr {{.*}} [[COERCE_CAST]], {{.*}}
  // CHECK:     [[EXTR_RVAL:%.*]]       = extractvalue {{.*}} [[RVAL]]{{.*}}
  // CHECK:     store {{.*}}[[EXTR_RVAL]]{{.*}}[[COERCE_CAST_PTR]]{{.*}}
  // CHECK:     [[COERCE_CAST:%.*]]     = bitcast %struct._NSRange* [[COERCE]]{{.*}}
  // CHECK:     [[SEL:%.*]]             = load i8*, i8** [[VALUE_SEL]]
  // CHECK:     [[RECV:%.*]]            = bitcast %struct._class_t* [[RECV_PTR]] to i8*
  // CHECK:     call {{.*objc_msgSend.*}}(i8* [[RECV]], i8* [[SEL]], i8* [[COERCE_CAST]], i8* {{.*}}[[RANGE_STR]]{{.*}})
  // CHECK:     call i8* @llvm.objc.retainAutoreleasedReturnValue
  NSValue *range_rvalue = @(getRange());
  // CHECK:     call void @llvm.objc.release
  // CHECK:     ret void
}

