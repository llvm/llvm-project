# REQUIRES: x86
# RUN: mkdir -p %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/test.o

# RUN: echo "_bar" > %t/order-file-1
# RUN: echo "_foo" >> %t/order-file-1
# RUN: echo "_main" >> %t/order-file-1
## _qux is marked as .alt_entry, so it should not create a new subsection and
## its contents should move with _bar to the start of the output despite the
## order file listing it at the end.
# RUN: echo "_qux" >> %t/order-file-1

## _bar and _baz point to the same address, so both order files should achieve
## the same result.
# RUN: echo "_baz" > %t/order-file-2
# RUN: echo "_foo" >> %t/order-file-2
# RUN: echo "_main" >> %t/order-file-2
# RUN: echo "_qux" >> %t/order-file-2

# RUN: lld -flavor darwinnew -arch x86_64 -o %t/test-1 %t/test.o -order_file %t/order-file-1
# RUN: llvm-objdump -d --no-show-raw-insn %t/test-1 | FileCheck %s
# RUN: lld -flavor darwinnew -arch x86_64 -o %t/test-2 %t/test.o -order_file %t/order-file-2
# RUN: llvm-objdump -d --no-show-raw-insn %t/test-2 | FileCheck %s
# CHECK-LABEL: Disassembly of section __TEXT,__text:
# CHECK:       <_bar>:
# CHECK-NEXT:    callq {{.*}} <_foo>
# CHECK-EMPTY:
# CHECK-NEXT:  <_qux>:
# CHECK-NEXT:    retq
# CHECK:       <_foo>:
# CHECK-NEXT:    retq
# CHECK:       <_main>:
# CHECK-NEXT:    callq {{.*}} <_bar>
# CHECK-NEXT:    movq $0, %rax
# CHECK-NEXT:    retq

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
