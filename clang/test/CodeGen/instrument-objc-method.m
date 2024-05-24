// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -debug-info-kind=standalone -emit-llvm -o - %s -finstrument-functions | FileCheck %s
// RUN: %clang_cc1 -disable-llvm-passes -triple x86_64-apple-darwin10 -debug-info-kind=standalone -emit-llvm -o - %s -finstrument-function-entry-bare | FileCheck -check-prefix=BARE %s

@interface ObjCClass
@end

@implementation ObjCClass

// CHECK: @"\01+[ObjCClass initialize]"
// CHECK: call void @__cyg_profile_func_enter
// CHECK: call void @__cyg_profile_func_exit
// BARE: @"\01+[ObjCClass initialize]"{{\(.*\)}} #0
+ (void)initialize {
}

// CHECK: @"\01+[ObjCClass load]"
// CHECK-NOT: call void @__cyg_profile_func_enter
// BARE: @"\01+[ObjCClass load]"{{\(.*\)}} #1
+ (void)load __attribute__((no_instrument_function)) {
}

// CHECK: @"\01-[ObjCClass dealloc]"
// CHECK-NOT: call void @__cyg_profile_func_enter
// BARE: @"\01-[ObjCClass dealloc]"{{\(.*\)}} #1
- (void)dealloc __attribute__((no_instrument_function)) {
}

// CHECK: declare void @__cyg_profile_func_enter(ptr, ptr)
// CHECK: declare void @__cyg_profile_func_exit(ptr, ptr)
// BARE-NOT: declare void @__cyg_profile_func_enter_bare
// BARE: attributes #0 = { {{.*}} "instrument-function-entry-inlined"="__cyg_profile_func_enter_bare"
// BARE-NOT: attributes #1 = { {{.*}} "__cyg_profile_func_enter_bare"
@end
