# REQUIRES: x86
## Check that section ordering follows from input file ordering.
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/1.s -o %t/1.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/2.s -o %t/2.o
# RUN: %lld -dylib %t/1.o %t/2.o -o %t/12
# RUN: %lld -dylib %t/2.o %t/1.o -o %t/21
# RUN: %lld -dylib %t/2.o %t/1.o -o %t/synth-section-order \
# RUN:     -add_empty_section __TEXT __objc_stubs \
# RUN:     -add_empty_section __TEXT __init_offsets \
# RUN:     -add_empty_section __TEXT __stubs \
# RUN:     -add_empty_section __TEXT __stub_helper \
# RUN:     -add_empty_section __TEXT __unwind_info \
# RUN:     -add_empty_section __TEXT __eh_frame \
# RUN:     -add_empty_section __DATA __objc_selrefs
# RUN: llvm-objdump --macho --section-headers %t/12 | FileCheck %s --check-prefix=CHECK-12
# RUN: llvm-objdump --macho --section-headers %t/21 | FileCheck %s --check-prefix=CHECK-21
# RUN: llvm-objdump --macho --section-headers %t/synth-section-order | FileCheck %s --check-prefix=CHECK-SYNTHETIC-ORDER

# CHECK-12:      __text
# CHECK-12-NEXT: foo
# CHECK-12-NEXT: bar
# CHECK-12-NEXT: __cstring

# CHECK-21:      __text
## `foo` always sorts next to `__text` since it's a code section
## and needs to be adjacent for arm64 thunk calculations
# CHECK-21-NEXT: foo
# CHECK-21-NEXT: __cstring
# CHECK-21-NEXT: bar

# CHECK-SYNTHETIC-ORDER:      __text
# CHECK-SYNTHETIC-ORDER-NEXT: foo
# CHECK-SYNTHETIC-ORDER-NEXT: __stubs
# CHECK-SYNTHETIC-ORDER-NEXT: __stub_helper
# CHECK-SYNTHETIC-ORDER-NEXT: __objc_stubs
# CHECK-SYNTHETIC-ORDER-NEXT: __init_offsets
# CHECK-SYNTHETIC-ORDER-NEXT: __cstring
# CHECK-SYNTHETIC-ORDER-NEXT: bar
# CHECK-SYNTHETIC-ORDER-NEXT: __unwind_info
# CHECK-SYNTHETIC-ORDER-NEXT: __eh_frame
# CHECK-SYNTHETIC-ORDER-NEXT: __objc_selrefs

#--- 1.s
.section __TEXT,foo
  .space 1
.section __TEXT,bar
  .space 1
.cstring
  .asciz ""

#--- 2.s
.cstring
  .asciz ""
.section __TEXT,bar
  .space 1
.section __TEXT,foo,regular,pure_instructions
  .space 1
