// RUN: %clang_cc1 -fmodules-ts -std=c++1z -triple=x86_64-linux-gnu -emit-module-interface %s -o %t.pcm
// RUN: %clang_cc1 -fmodules-ts -std=c++1z -triple=x86_64-linux-gnu %t.pcm -emit-llvm -o - | FileCheck %s

module FooBar;

export {
  // CHECK-LABEL: define i32 @_Z1fv(
  int f() { return 0; }
}

// FIXME: Emit global variables and their initializers with this TU.
// Emit an initialization function that other TUs can call, with guard variable.

// FIXME: Mangle non-exported symbols so they don't collide with
// non-exported symbols from other modules?

// FIXME: Formally-internal-linkage symbols that are used from an exported
// symbol need a mangled name and external linkage.

// FIXME: const-qualified variables don't have implicit internal linkage when owned by a module.
