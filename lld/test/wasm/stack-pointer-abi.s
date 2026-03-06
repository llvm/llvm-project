# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t/use.o %t/use.s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t/disallow.o %t/disallow.s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t/start.o %t/start.s
# RUN: wasm-ld --component-model-thread-context -o %t/component-model.wasm %t/use.o %t/start.o
# RUN: obj2yaml %t/component-model.wasm | FileCheck %s --check-prefix=COMPONENT-MODEL
# RUN: wasm-ld -o %t/global.wasm %t/disallow.o %t/start.o
# RUN: obj2yaml %t/global.wasm | FileCheck %s --check-prefix=GLOBAL

#--- start.s
  .globl  _start
_start:
  .functype _start () -> ()
  end_function

#--- disallow.s
.section  .custom_section.target_features,"",@
  .int8 1
  .int8 45
  .int8 30
  .ascii  "component-model-thread-context"

#--- use.s

.section  .custom_section.target_features,"",@
  .int8 1
  .int8 43
  .int8 30
  .ascii  "component-model-thread-context"

# COMPONENT-MODEL: Name: __init_stack_pointer
# GLOBAL: Name: __stack_pointer
