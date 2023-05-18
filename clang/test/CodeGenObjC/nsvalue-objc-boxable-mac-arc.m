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

// CHECK-LABEL: define{{.*}} void @doRange()
void doRange(void) {
  // CHECK:      [[LOCAL_VAR:%.*]]  = alloca %struct._NSRange{{.*}}
  // CHECK:      [[TEMP_VAR:%.*]]   = alloca %struct._NSRange{{.*}}
  // CHECK:      [[RECV_PTR:%.*]]   = load {{.*}} [[NSVALUE]]
  // CHECK:      call void @llvm.memcpy{{.*}} [[TEMP_VAR]]{{.*}} [[LOCAL_VAR]]{{.*}}
  // CHECK:      [[SEL:%.*]]        = load ptr, ptr [[VALUE_SEL]]
  NSRange ns_range = { .location = 0, .length = 42 };
  // CHECK:      call {{.*objc_msgSend.*}}(ptr noundef [[RECV_PTR]], ptr noundef [[SEL]], ptr noundef [[TEMP_VAR]], ptr {{.*}}[[RANGE_STR]]{{.*}}) [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  NSValue *range = @(ns_range);
  // CHECK:      call void @llvm.objc.release
  // CHECK:      ret void
}

// CHECK-LABEL: define{{.*}} void @doPoint()
void doPoint(void) {
  // CHECK:      [[LOCAL_VAR:%.*]]  = alloca %struct._NSPoint{{.*}}
  // CHECK:      [[TEMP_VAR:%.*]]   = alloca %struct._NSPoint{{.*}}
  // CHECK:      [[RECV_PTR:%.*]]   = load {{.*}} [[NSVALUE]]
  // CHECK:      call void @llvm.memcpy{{.*}} [[TEMP_VAR]]{{.*}} [[LOCAL_VAR]]{{.*}}
  // CHECK:      [[SEL:%.*]]        = load ptr, ptr [[VALUE_SEL]]
  NSPoint ns_point = { .x = 42, .y = 24 };
  // CHECK:      call {{.*objc_msgSend.*}}(ptr noundef [[RECV_PTR]], ptr noundef [[SEL]], ptr noundef [[TEMP_VAR]], ptr {{.*}}[[POINT_STR]]{{.*}}) [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  NSValue *point = @(ns_point);
  // CHECK:      call void @llvm.objc.release
  // CHECK:      ret void
}

// CHECK-LABEL: define{{.*}} void @doSize()
void doSize(void) {
  // CHECK:      [[LOCAL_VAR:%.*]]  = alloca %struct._NSSize{{.*}}
  // CHECK:      [[TEMP_VAR:%.*]]   = alloca %struct._NSSize{{.*}}
  // CHECK:      [[RECV_PTR:%.*]]   = load {{.*}} [[NSVALUE]]
  // CHECK:      call void @llvm.memcpy{{.*}} [[TEMP_VAR]]{{.*}} [[LOCAL_VAR]]{{.*}}
  // CHECK:      [[SEL:%.*]]        = load ptr, ptr [[VALUE_SEL]]
  NSSize ns_size = { .width = 42, .height = 24 };
  // CHECK:      call {{.*objc_msgSend.*}}(ptr noundef [[RECV_PTR]], ptr noundef [[SEL]], ptr noundef [[TEMP_VAR]], ptr {{.*}}[[SIZE_STR]]{{.*}}) [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  NSValue *size = @(ns_size);
  // CHECK:      call void @llvm.objc.release
  // CHECK:      ret void
}

// CHECK-LABEL: define{{.*}} void @doRect()
void doRect(void) {
  // CHECK:      [[LOCAL_VAR:%.*]]  = alloca %struct._NSRect{{.*}}
  // CHECK:      [[TEMP_VAR:%.*]]   = alloca %struct._NSRect{{.*}}
  // CHECK:      [[RECV_PTR:%.*]]   = load {{.*}} [[NSVALUE]]
  // CHECK:      call void @llvm.memcpy{{.*}} [[TEMP_VAR]]{{.*}} [[LOCAL_VAR]]{{.*}}
  // CHECK:      [[SEL:%.*]]        = load ptr, ptr [[VALUE_SEL]]
  NSPoint ns_point = { .x = 42, .y = 24 };
  NSSize ns_size = { .width = 42, .height = 24 };
  NSRect ns_rect = { .origin = ns_point, .size = ns_size };
  // CHECK:      call {{.*objc_msgSend.*}}(ptr noundef [[RECV_PTR]], ptr noundef [[SEL]], ptr noundef [[TEMP_VAR]], ptr{{.*}}[[RECT_STR]]{{.*}}) [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  NSValue *rect = @(ns_rect);
  // CHECK:      call void @llvm.objc.release
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
  // CHECK:      call {{.*objc_msgSend.*}}(ptr noundef [[RECV_PTR]], ptr noundef [[SEL]], ptr noundef [[TEMP_VAR]], ptr{{.*}}[[EDGE_STR]]{{.*}}) [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  NSValue *edge_insets = @(ns_edge_insets);
  // CHECK:      call void @llvm.objc.release
  // CHECK:      ret void
}

// CHECK-LABEL: define{{.*}} void @doRangeRValue() 
void doRangeRValue(void) {
  // CHECK:     [[COERCE:%.*]]          = alloca %struct._NSRange{{.*}}
  // CHECK:     [[RECV_PTR:%.*]]        = load {{.*}} [[NSVALUE]]
  // CHECK:     [[RVAL:%.*]]            = call {{.*}} @getRange()
  // CHECK:     [[COERCE_VAR_PTR:%.*]] = getelementptr {{.*}} [[COERCE]], {{.*}}
  // CHECK:     [[EXTR_RVAL:%.*]]       = extractvalue {{.*}} [[RVAL]]{{.*}}
  // CHECK:     store {{.*}}[[EXTR_RVAL]]{{.*}}[[COERCE_VAR_PTR]]{{.*}}
  // CHECK:     [[SEL:%.*]]             = load ptr, ptr [[VALUE_SEL]]
  // CHECK:     call {{.*objc_msgSend.*}}(ptr noundef [[RECV_PTR]], ptr noundef [[SEL]], ptr noundef [[COERCE]], ptr {{.*}}[[RANGE_STR]]{{.*}}) [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  NSValue *range_rvalue = @(getRange());
  // CHECK:     call void @llvm.objc.release
  // CHECK:     ret void
}

