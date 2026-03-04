// RUN: %clang_cc1 -disable-llvm-passes -triple x86_64-apple-darwin10 -debug-info-kind=standalone -emit-llvm -o - %s -finstrument-functions | FileCheck --check-prefix=PREINLINE --implicit-check-not="__cyg_profile_func_enter" %s
// RUN: %clang_cc1 -disable-llvm-passes -triple x86_64-apple-darwin10 -debug-info-kind=standalone -emit-llvm -o - %s -finstrument-function-entry-bare | FileCheck --check-prefix=BARE --implicit-check-not="__cyg_profile_func_enter" %s

@interface ObjCClass
@end

@implementation ObjCClass

// PREINLINE: define {{.*}}@"\01+[ObjCClass initialize]"{{\(.*\)}} #[[#ATTR:]]
// BARE: define {{.*}}@"\01+[ObjCClass initialize]"{{\(.*\)}} #[[#ATTR:]]
+ (void)initialize {
}

+ (void)load __attribute__((no_instrument_function)) {
}

- (void)dealloc __attribute__((no_instrument_function)) {
}

// PREINLINE: attributes #[[#ATTR]] =
// PREINLINE-SAME: "instrument-function-entry"="__cyg_profile_func_enter"
// BARE: attributes #[[#ATTR]] =
// BARE-SAME: "instrument-function-entry-inlined"="__cyg_profile_func_enter_bare"
@end
