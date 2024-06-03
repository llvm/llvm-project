// RUN: %clang_cc1 -disable-llvm-passes -triple x86_64-apple-darwin10 -debug-info-kind=standalone -emit-llvm -o - %s -finstrument-functions | FileCheck -check-prefix=PREINLINE %s
// RUN: %clang_cc1 -disable-llvm-passes -triple x86_64-apple-darwin10 -debug-info-kind=standalone -emit-llvm -o - %s -finstrument-function-entry-bare | FileCheck -check-prefix=BARE %s

@interface ObjCClass
@end

@implementation ObjCClass

// PREINLINE: @"\01+[ObjCClass initialize]"{{\(.*\)}} #0
// BARE: @"\01+[ObjCClass initialize]"{{\(.*\)}} #0
+ (void)initialize {
}

// PREINLINE: declare void @llvm.dbg.declare(metadata, metadata, metadata) #1
// BARE: @"\01+[ObjCClass load]"{{\(.*\)}} #2
+ (void)load __attribute__((no_instrument_function)) {
}

// PREINLINE: @"\01-[ObjCClass dealloc]"{{\(.*\)}} #2
// BARE: @"\01-[ObjCClass dealloc]"{{\(.*\)}} #2
- (void)dealloc __attribute__((no_instrument_function)) {
}

// PREINLINE: attributes #0 = { {{.*}}"instrument-function-entry"="__cyg_profile_func_enter"
// PREINLINE-NOT: attributes #0 = { {{.*}}"instrument-function-entry"="__cyg_profile_func_enter_bare"
// PREINLINE-NOT: attributes #2 = { {{.*}}"__cyg_profile_func_enter"
// BARE: attributes #0 = { {{.*}}"instrument-function-entry-inlined"="__cyg_profile_func_enter_bare"
// BARE-NOT: attributes #0 = { {{.*}}"__cyg_profile_func_enter"
// BARE-NOT: attributes #2 = { {{.*}}"__cyg_profile_func_enter_bare"
@end
