// REQUIRES: spirv-registered-target
//
// Verify that __spirv_event_t round trips through a C++20 module.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t

// RUN: %clang_cc1 -std=c++20 -triple spirv64 %t/EventMod.cppm -emit-module-interface -o %t/EventMod.pcm
// RUN: %clang_cc1 -std=c++20 -triple spirv64 %t/UseEventMod.cpp -fmodule-file=EventMod=%t/EventMod.pcm -ast-dump-all | FileCheck %t/UseEventMod.cpp

// expected-no-diagnostics

//--- EventMod.cppm
export module EventMod;
export __spirv_event_t getEvent();
export void useEvent(__spirv_event_t e);

//--- UseEventMod.cpp
import EventMod;
// Check the declarations deserialized from the module (not the uses in test()).
// CHECK: FunctionDecl 0x{{.*}} imported in EventMod {{.*}} getEvent '__spirv_event_t ()'
// CHECK: FunctionDecl 0x{{.*}} imported in EventMod {{.*}} useEvent 'void (__spirv_event_t)'
// CHECK: ParmVarDecl 0x{{.*}} imported in EventMod {{.*}} e '__spirv_event_t'

void test() {
  __spirv_event_t e = getEvent();
  useEvent(e);
}
