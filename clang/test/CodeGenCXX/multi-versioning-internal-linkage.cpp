// RUN: %clang_cc1 -no-opaque-pointers -std=c++11 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=LINUX
// RUN: %clang_cc1 -no-opaque-pointers -std=c++11 -triple x86_64-windows-pc -emit-llvm %s -o - | FileCheck %s --check-prefix=WINDOWS

__attribute__((target("default"))) static int static_target() {
  return 0;
}
__attribute__((target("arch=sandybridge"))) static int static_target() {
  return 1;
}

__attribute__((cpu_dispatch(atom, generic))) static int static_dispatch() {}

namespace {
int __attribute__((target("default"))) anon_target() {
  return 0;
}
int __attribute__((target("arch=sandybridge"))) anon_target() {
  return 1;
}
__attribute__((cpu_dispatch(atom, generic))) static int anon_dispatch() {}
}



int usage() {
  return static_target() + anon_target() + static_dispatch() + anon_dispatch();
}

// Ensure that these are all 'internal'. Windows doesn't have ifuncs or aliases,
// since Windows doesn't support ifuncs.
// aliases:
// LINUX: @_ZL15static_dispatchv = internal alias i32 (), i32 ()* @_ZL15static_dispatchv.ifunc
// LINUX: @_ZN12_GLOBAL__N_113anon_dispatchEv = internal alias i32 (), i32 ()* @_ZN12_GLOBAL__N_113anon_dispatchEv.ifunc

// ifuncs:
// LINUX: @_ZL15static_dispatchv.ifunc = internal ifunc i32 (), i32 ()* ()* @_ZL15static_dispatchv.resolver
// LINUX: @_ZN12_GLOBAL__N_113anon_dispatchEv.ifunc = internal ifunc i32 (), i32 ()* ()* @_ZN12_GLOBAL__N_113anon_dispatchEv.resolver
// LINUX: @_ZL13static_targetv.ifunc = internal ifunc i32 (), i32 ()* ()* @_ZL13static_targetv.resolver
// LINUX: @_ZN12_GLOBAL__N_111anon_targetEv.ifunc = internal ifunc i32 (), i32 ()* ()* @_ZN12_GLOBAL__N_111anon_targetEv.resolver

// resolvers:
// LINUX: define internal i32 ()* @_ZL15static_dispatchv.resolver()
// WINDOWS: define internal i32 @"?static_dispatch@@YAHXZ"()
// LINUX: define internal i32 ()* @_ZN12_GLOBAL__N_113anon_dispatchEv.resolver()
// WINDOWS: define internal i32 @"?anon_dispatch{{.*}}@@YAHXZ"()
// LINUX: define internal i32 ()* @_ZL13static_targetv.resolver()
// WINDOWS: define internal i32 @"?static_target@@YAHXZ.resolver"()
// LINUX: define internal i32 ()* @_ZN12_GLOBAL__N_111anon_targetEv.resolver()
// WINDOWS: define internal i32 @"?anon_target{{.*}}@@YAHXZ.resolver"()
