# Test that linking object files with mismatched thread context ABIs fails with an error.

# RUN: split-file %s %t

# Test that the presence of an import of __stack_pointer from the env module is treated 
# as an indication that the global thread context ABI is being used, even if the
# component-model-thread-context feature is not disallowed.

# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t/start.o %t/start.s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t/stack-pointer.o %t/stack-pointer.s
# RUN: not wasm-ld %t/start.o %t/stack-pointer.o -o %t/fail.wasm 2>&1 | FileCheck %s

# Test that explicitly disallowing the component-model-thread-context feature causes linking to fail 
# with an error when other files use the feature.

# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t/disallow.o %t/disallow.s
# RUN: not wasm-ld %t/start.o %t/disallow.o -o %t/fail.wasm 2>&1 | FileCheck %s

# CHECK: error: thread context ABI mismatch: {{.*}} disallows component-model-thread-context but other files use it 

#--- start.s
.globl _start
_start:
  .functype _start () -> ()
  end_function

# Mark the feature as USED
.section  .custom_section.target_features,"",@
  .int8 1
  .int8 43
  .int8 30
  .ascii  "component-model-thread-context"

#--- stack-pointer.s
.globaltype __stack_pointer, i32

.globl  _start
_start:
  .functype _start () -> (i32)
  global.get __stack_pointer
  i32.const 16
  i32.sub
  drop
  i32.const 0
  end_function

#--- disallow.s
.section  .custom_section.target_features,"",@
  .int8 1
  .int8 45
  .int8 30
  .ascii  "component-model-thread-context"
