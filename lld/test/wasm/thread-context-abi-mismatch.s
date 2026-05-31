# Test that linking object files with mismatched thread context ABIs fails with an error.
# The presence of an import of __stack_pointer from the env module should be treated 
# as an indication that the global thread context ABI is being used.

# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: not wasm-ld --libcall-thread-context %t.o -o %t.wasm 2>&1 | FileCheck %s
# RUN: not wasm-ld --cooperative-multithreading %t.o -o %t.wasm 2>&1 | FileCheck %s

# CHECK: object file uses globals for thread context, but --libcall-thread-context or --cooperative-multithreading was specified
.globl _start
_start:
  .functype _start () -> ()
  end_function

.globaltype __stack_pointer, i32

.globl use_stack_pointer
use_stack_pointer:
  .functype use_stack_pointer () -> ()
  global.get __stack_pointer
  drop
  end_function
