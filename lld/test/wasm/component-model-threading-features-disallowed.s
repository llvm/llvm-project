# Test that objects with component-model-thread-context feature marked as DISALLOWED
# cannot link with --component-model-thread-context flag

# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t-without.o %s
# RUN: wasm-ld %t-without.o -o %t.wasm
# RUN: not wasm-ld --component-model-thread-context %t-without.o -o %t2.wasm 2>&1 | FileCheck %s

# CHECK: error: --component-model-thread-context is disallowed by {{.*}} because it was not compiled with the 'component-model-thread-context' feature.

.globl _start
_start:
  .functype _start () -> ()
  end_function

# Mark the feature as DISALLOWED (0x2d = '-' = WASM_FEATURE_PREFIX_DISALLOWED)
.section  .custom_section.target_features,"",@
  .int8 1
  .int8 45
  .int8 30
  .ascii  "component-model-thread-context"
