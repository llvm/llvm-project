// RUN: %clang_cc1 -no-opaque-pointers -S -triple x86_64-apple-darwin10 -debug-info-kind=standalone -emit-llvm -o - %s -finstrument-functions | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -S -triple x86_64-apple-darwin10 -debug-info-kind=standalone -emit-llvm -o - %s -finstrument-function-entry-bare | FileCheck -check-prefix=BARE %s

@interface ObjCClass
@end

@implementation ObjCClass

// CHECK: @"\01+[ObjCClass initialize]"
// CHECK: call void @__cyg_profile_func_enter
// CHECK: call void @__cyg_profile_func_exit
// BARE: @"\01+[ObjCClass initialize]"
// BARE: call void @__cyg_profile_func_enter
+ (void)initialize {
}

// CHECK: @"\01+[ObjCClass load]"
// CHECK-NOT: call void @__cyg_profile_func_enter
// BARE: @"\01+[ObjCClass load]"
// BARE-NOT: call void @__cyg_profile_func_enter
+ (void)load __attribute__((no_instrument_function)) {
}

// CHECK: @"\01-[ObjCClass dealloc]"
// CHECK-NOT: call void @__cyg_profile_func_enter
// BARE: @"\01-[ObjCClass dealloc]"
// BARE-NOT: call void @__cyg_profile_func_enter
- (void)dealloc __attribute__((no_instrument_function)) {
}

// CHECK: declare void @__cyg_profile_func_enter(i8*, i8*)
// CHECK: declare void @__cyg_profile_func_exit(i8*, i8*)
// BARE: declare void @__cyg_profile_func_enter_bare
@end
