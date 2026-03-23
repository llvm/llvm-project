//===-- thread_context.c - Provide access to stack pointer and TLS base ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements __wasm_get_{stack_pointer,tls_base} and
// __wasm_set_{stack_pointer,tls_base}, which are used for accessing the
// component model thread context.
//
//===----------------------------------------------------------------------===//

#ifdef __wasm_component_model_thread_context__

// We define these function as naked functions with inline assembly because:
// 1. Defining them as regular C functions would cause infinite recursion in
//    the prologue/epilogue code that accesses the stack pointer.
// 2. The compiler-rt build system doesn't pass the target triple when compiling
//    assembly files, and making it do so would require a non-trivial amount of
//    build system changes.

__attribute__((naked)) void *__wasm_get_stack_pointer(void) {
  __asm__ volatile(
      ".functype __wasm_component_model_builtin_context_get_0 () -> (i32)\n"
      ".import_module __wasm_component_model_builtin_context_get_0, \"$root\"\n"
      ".import_name __wasm_component_model_builtin_context_get_0, "
      "\"[context-get-0]\"\n"
      "call __wasm_component_model_builtin_context_get_0");
}

__attribute__((naked)) void __wasm_set_stack_pointer(void *ptr) {
  __asm__ volatile(
      ".functype __wasm_component_model_builtin_context_set_0 (i32) -> ()\n"
      ".import_module __wasm_component_model_builtin_context_set_0, \"$root\"\n"
      ".import_name __wasm_component_model_builtin_context_set_0, "
      "\"[context-set-0]\"\n"
      "local.get 0\n"
      "call __wasm_component_model_builtin_context_set_0");
}

__attribute__((naked)) void *__wasm_get_tls_base(void) {
  __asm__ volatile(
      ".functype __wasm_component_model_builtin_context_get_1 () -> (i32)\n"
      ".import_module __wasm_component_model_builtin_context_get_1, \"$root\"\n"
      ".import_name __wasm_component_model_builtin_context_get_1, "
      "\"[context-get-1]\"\n"
      "call __wasm_component_model_builtin_context_get_1");
}
__attribute__((naked)) void __wasm_set_tls_base(void *ptr) {
  __asm__ volatile(
      ".functype __wasm_component_model_builtin_context_set_1 (i32) -> ()\n"
      ".import_module __wasm_component_model_builtin_context_set_1, \"$root\"\n"
      ".import_name __wasm_component_model_builtin_context_set_1, "
      "\"[context-set-1]\"\n"
      "local.get 0\n"
      "call __wasm_component_model_builtin_context_set_1");
}

#endif
