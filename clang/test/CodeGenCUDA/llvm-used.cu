// RUN: %clang_cc1 -emit-llvm %s -o - -fcuda-is-device -triple nvptx64-unknown-unknown | FileCheck %s
// RUN: %clang_cc1 -emit-llvm %s -o - -fcuda-is-device -triple amdgcn-amd-amdhsa | FileCheck %s --check-prefix=NOASCAST


// Make sure we emit the proper addrspacecast for llvm.used iff necessary.
// PR22383 exposed an issue where we were generating a bitcast instead of an
// addrspacecast.

// CHECK: @llvm.compiler.used = appending global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @a to ptr)], section "llvm.metadata"
// NOASCAST: @llvm.compiler.used = appending addrspace(1) global [1 x ptr addrspace(1)] [ptr addrspace(1) @a], section "llvm.metadata"
__attribute__((device)) __attribute__((__used__)) int a[] = {};
