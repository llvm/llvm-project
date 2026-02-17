# Test that objects with component-model-thread-context feature marked as USED
# can only link with --component-model-thread-context flag

# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t-with.o %s
# RUN: wasm-ld --component-model-thread-context %t-with.o -o %t.wasm
# RUN: not wasm-ld %t-with.o -o %t2.wasm 2>&1 | FileCheck %s

# CHECK: error: component-model-thread-context feature used by {{.*}} but --component-model-thread-context not specified. 

.globl _start
_start:
  .functype _start () -> ()
  end_function

# Mark the feature as USED (0x2b = '+' = WASM_FEATURE_PREFIX_USED)
.section  .custom_section.target_features,"",@
  .int8 1
  .int8 43
  .int8 30
  .ascii  "component-model-thread-context"
