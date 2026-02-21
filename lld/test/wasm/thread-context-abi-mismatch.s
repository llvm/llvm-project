# Test that linking object files with mismatched thread context ABIs fails with an error.

# Test that the presence of an import of __stack_pointer from the env module is treated 
# as an indication that the global thread context ABI is being used, even if the
# component-model-thread-context feature is not disallowed.

# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t-with.o %s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t-without.o %S/Inputs/stack-pointer.s
# RUN: not wasm-ld %t-with.o %t-without.o -o %t.wasm 2>&1 | FileCheck %s

# Test that explicitly disallowing the component-model-thread-context feature causes linking to fail 
# with an error when other files use the feature.

# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t-disallow.o %S/Inputs/disallow-component-model-thread-context.s
# RUN: not wasm-ld %t-with.o %t-disallow.o -o %t.wasm 2>&1 | FileCheck %s

# CHECK: error: thread context ABI mismatch: {{.*}} disallows component-model-thread-context but other files use it 
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
