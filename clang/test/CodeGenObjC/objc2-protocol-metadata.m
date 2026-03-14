// RUN: %clang_cc1 -triple x86_64-apple-macosx10.10 -emit-llvm -o - %s | FileCheck %s

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

// CHECK: %struct._protocol_t = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr }
