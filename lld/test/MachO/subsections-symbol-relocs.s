# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o

# RUN: %lld -o %t/test-1 %t/test.o -order_file %t/order-file-1
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t/test-1 | FileCheck %s
# RUN: %lld -o %t/test-2 %t/test.o -order_file %t/order-file-2
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t/test-2 | FileCheck %s
# CHECK-LABEL: Disassembly of section __TEXT,__text:
# CHECK:       <_ba{{r|z}}>:
# CHECK-NEXT:    callq {{.*}} <_foo>
# CHECK-EMPTY:
# CHECK-NEXT:  <_qux>:
# CHECK-NEXT:    retq
# CHECK:       <_foo>:
# CHECK-NEXT:    retq
# CHECK:       <_main>:
# CHECK-NEXT:    callq {{.*}} <_ba{{r|z}}>
# CHECK-NEXT:    movq $0, %rax
# CHECK-NEXT:    retq

#--- order-file-1
_bar
_foo
_main
## _qux is marked as .alt_entry, so it should not create a new subsection and
## its contents should move with _bar to the start of the output despite the
## order file listing it at the end.
_qux

#--- order-file-2
## _bar and _baz point to the same address, so both order files should achieve
## the same result.
_baz
_foo
_main
_qux

#--- test.s
.text
.globl _main, _foo, _bar, _qux
.alt_entry _qux

_foo:
  retq

_main:
  callq _bar
  movq $0, %rax
  retq

_bar:
_baz:
  callq _foo
_qux:
  retq

.subsections_via_symbols
