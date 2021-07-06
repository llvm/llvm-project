# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: %lld -dylib %t/foo.o -o %t/libfoo.dylib
# RUN: %lld -lSystem %t/test.o %t/libfoo.dylib -o %t/test

## Make sure we emit exactly one BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM per
## symbol.
# RUN: obj2yaml %t/test | FileCheck %s --implicit-check-not BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM

# CHECK:      Opcode:          BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM
# CHECK-NEXT: Imm:             0
# CHECK-NEXT: Symbol:          _foo

# CHECK:      Opcode:          BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM
# CHECK-NEXT: Imm:             0
# CHECK-NEXT: Symbol:          _bar

# RUN: llvm-objdump --macho --bind %t/test | FileCheck %s --check-prefix=BIND
# BIND:       Bind table:
# BIND-NEXT:  segment  section address type    addend dylib   symbol
# BIND-NEXT:  __DATA   __data  {{.*}}  pointer      0 libfoo  _foo
# BIND-NEXT:  __DATA   __data  {{.*}}  pointer      0 libfoo  _foo
# BIND-NEXT:  __DATA   __data  {{.*}}  pointer      0 libfoo  _bar
# BIND-NEXT:  __DATA   __data  {{.*}}  pointer      0 libfoo  _bar
# BIND-EMPTY:

#--- foo.s
.globl _foo, _bar
_foo:
  .space 4
_bar:
  .space 4

#--- test.s
.data
.quad _foo
.quad _bar
.quad _foo
.quad _bar

.globl _main
.text
_main:
