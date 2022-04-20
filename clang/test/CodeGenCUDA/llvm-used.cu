// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm %s -o - -fcuda-is-device -triple nvptx64-unknown-unknown | FileCheck %s


// Make sure we emit the proper addrspacecast for llvm.used.  PR22383 exposed an
// issue where we were generating a bitcast instead of an addrspacecast.

// CHECK: @llvm.compiler.used = appending global [1 x i8*] [i8* addrspacecast (i8 addrspace(1)* bitcast ([0 x i32] addrspace(1)* @a to i8 addrspace(1)*) to i8*)], section "llvm.metadata"
__attribute__((device)) __attribute__((__used__)) int a[] = {};
