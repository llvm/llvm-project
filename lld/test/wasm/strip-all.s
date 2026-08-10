# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.start.o
# RUN: wasm-ld --strip-all -o %t.wasm %t.start.o
# RUN: obj2yaml %t.wasm | FileCheck %s
#
## Test alias -s
# RUN: wasm-ld -s -o %t2.wasm %t.start.o
# RUN: obj2yaml %t2.wasm | FileCheck %s
#
## Check that there is no name section
# CHECK-NOT:   Name:    name
# CHECK-NOT:   Name:    target_features
#
## Test --keep-section=name preserver the name section
# RUN: wasm-ld --strip-all --keep-section=name -o %t3.wasm %t.start.o
# RUN: obj2yaml %t3.wasm | FileCheck --check-prefix=CHECK-NAME %s
#
# CHECK-NAME:   Name:    name
# CHECK-NAME-NOT:   Name:    target_features
#
## Test --keep-section can be specified more than once
# RUN: wasm-ld --strip-all --keep-section=name --keep-section=target_features -o %t4.wasm %t.start.o
# RUN: obj2yaml %t4.wasm | FileCheck --check-prefix=CHECK-FEATURES %s
#
# CHECK-FEATURES:   Name:    name
# CHECK-FEATURES:   Name:    target_features

  .globl  _start
_start:
  .functype _start () -> ()
  end_function

.section .custom_section.target_features,"",@
.int8 1
.int8 43
.int8 15
.ascii "mutable-globals"
