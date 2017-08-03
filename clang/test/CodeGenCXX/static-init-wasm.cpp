// RUN: %clang_cc1 -emit-llvm -triple=wasm32-unknown-unknown -o - %s \
// RUN:   | FileCheck %s -check-prefix=WEBASSEMBLY32
// RUN: %clang_cc1 -emit-llvm -triple=wasm64-unknown-unknown -o - %s \
// RUN:   | FileCheck %s -check-prefix=WEBASSEMBLY64

// Test that we don't create common blocks.
int tentative;
// WEBASSEMBLY32: @tentative = global i32 0, align 4
// WEBASSEMBLY64: @tentative = global i32 0, align 4

// Test that WebAssembly uses the ARM-style ABI in which the static
// variable's guard variable is tested via "load i8 and test the
// bottom bit" rather than the Itanium/x86 ABI which uses "load i8
// and compare with zero".
int f();
void g() {
  static int a = f();
}
// WEBASSEMBLY32-LABEL: @_Z1gv()
// WEBASSEMBLY32:       %[[R0:.+]] = load atomic i8, i8* bitcast (i32* @_ZGVZ1gvE1a to i8*) acquire, align 4
// WEBASSEMBLY32-NEXT:  %[[R1:.+]] = and i8 %[[R0]], 1
// WEBASSEMBLY32-NEXT:  %[[R2:.+]] = icmp eq i8 %[[R1]], 0
// WEBASSEMBLY32-NEXT:  br i1 %[[R2]], label %[[CHECK:.+]], label %[[END:.+]]
// WEBASSEMBLY32:       [[CHECK]]
// WEBASSEMBLY32:       call i32 @__cxa_guard_acquire
// WEBASSEMBLY32:       [[END]]
// WEBASSEMBLY32:       call void @__cxa_guard_release
//
// WEBASSEMBLY64-LABEL: @_Z1gv()
// WEBASSEMBLY64:       %[[R0:.+]] = load atomic i8, i8* bitcast (i64* @_ZGVZ1gvE1a to i8*) acquire, align 8
// WEBASSEMBLY64-NEXT:  %[[R1:.+]] = and i8 %[[R0]], 1
// WEBASSEMBLY64-NEXT:  %[[R2:.+]] = icmp eq i8 %[[R1]], 0
// WEBASSEMBLY64-NEXT:  br i1 %[[R2]], label %[[CHECK:.+]], label %[[END:.+]]
// WEBASSEMBLY64:       [[CHECK]]
// WEBASSEMBLY64:       call i32 @__cxa_guard_acquire
// WEBASSEMBLY64:       [[END]]
// WEBASSEMBLY64:       call void @__cxa_guard_release

// Test various aspects of static constructor calls.
struct A {
    A();
};

A theA;

// WEBASSEMBLY32: define internal void @__cxx_global_var_init() #3 section ".text.__startup" {
// WEBASSEMBLY32: call %struct.A* @_ZN1AC1Ev(%struct.A* @theA)
// WEBASSEMBLY32: define internal void @_GLOBAL__sub_I_static_init_wasm.cpp() #3 section ".text.__startup" {
// WEBASSEMBLY32: call void @__cxx_global_var_init()
//
// WEBASSEMBLY64: define internal void @__cxx_global_var_init() #3 section ".text.__startup" {
// WEBASSEMBLY64: call %struct.A* @_ZN1AC1Ev(%struct.A* @theA)
// WEBASSEMBLY64: define internal void @_GLOBAL__sub_I_static_init_wasm.cpp() #3 section ".text.__startup" {
// WEBASSEMBLY64: call void @__cxx_global_var_init()
