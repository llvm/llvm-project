// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-macosx10.10 -emit-llvm -o - %s | FileCheck %s
// rdar://20286356

@protocol P1
- InstP;
+ ClsP;
@end

@interface INTF <P1>
@end

@implementation INTF
- InstP { return 0; }
+ ClsP  { return 0; }
@end

// CHECK: %struct._protocol_t = type { i8*, i8*, %struct._objc_protocol_list*, %struct.__method_list_t*, %struct.__method_list_t*, %struct.__method_list_t*, %struct.__method_list_t*, %struct._prop_list_t*, i32, i32, i8**, i8*, %struct._prop_list_t* }
