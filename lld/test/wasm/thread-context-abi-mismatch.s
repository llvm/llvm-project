# Test that linking object files with mismatched thread context ABIs fails with an error.

# RUN: split-file %s %t

# Test that the presence of an import of __stack_pointer from the env module is treated 
# as an indication that the global thread context ABI is being used.

# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t/start.o %t/start.s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t/stack-pointer.o %t/stack-pointer.s
# RUN: not wasm-ld --libcall-thread-context %t/start.o %t/stack-pointer.o -o %t/fail.wasm 2>&1 | FileCheck %s

# CHECK: stack-pointer.o: object file uses globals for thread context, but --libcall-thread-context was specified

#--- start.s
.globl _start
_start:
  .functype _start () -> ()
  end_function

#--- stack-pointer.s
.globaltype __stack_pointer, i32

.globl use_stack_pointer
use_stack_pointer:
  .functype use_stack_pointer () -> ()
  global.get __stack_pointer
  drop
  end_function
