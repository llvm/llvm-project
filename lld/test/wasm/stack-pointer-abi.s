# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t/use.o %t/use.s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t/disallow.o %t/disallow.s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t/start.o %t/start.s
# RUN: wasm-ld -o %t/libcall.wasm %t/use.o %t/start.o
# RUN: obj2yaml %t/libcall.wasm | FileCheck %s --check-prefix=LIBCALL
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
  .int8 22
  .ascii  "libcall-thread-context"

#--- use.s

.section  .custom_section.target_features,"",@
  .int8 1
  .int8 43
  .int8 2
  .ascii  "libcall-thread-context"

# LIBCALL: Name: __init_stack_pointer
# GLOBAL: Name: __stack_pointer
