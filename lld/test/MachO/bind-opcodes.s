# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: %lld -O2 -dylib %t/foo.o -o %t/libfoo.dylib
# RUN: %lld -O2 -lSystem %t/test.o %t/libfoo.dylib -o %t/test

## Test:
## 1/ We emit exactly one BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM per symbol.
## 2/ Combine BIND_OPCODE_DO_BIND and BIND_OPCODE_ADD_ADDR_ULEB pairs.
## 3/ Compact BIND_OPCODE_DO_BIND_ADD_ADDR_ULEB
# RUN: obj2yaml %t/test | FileCheck %s

# CHECK:      BindOpcodes:
# CHECK-NEXT:   Opcode:          BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM
# CHECK-NEXT:   Imm:             0
# CHECK-NEXT:   Symbol:          _foo
# CHECK-NEXT:   Opcode:          BIND_OPCODE_SET_TYPE_IMM
# CHECK-NEXT:   Imm:             1
# CHECK-NEXT:   Symbol:          ''
# CHECK-NEXT:   Opcode:          BIND_OPCODE_SET_DYLIB_ORDINAL_IMM
# CHECK-NEXT:   Imm:             2
# CHECK-NEXT:   Symbol:          ''
# CHECK-NEXT:   Opcode:          BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB
# CHECK-NEXT:   Imm:             2
# CHECK-NEXT:   ULEBExtraData:   [ 0x0 ]
# CHECK-NEXT:   Symbol:          ''
# CHECK-NEXT:   Opcode:          BIND_OPCODE_DO_BIND_ULEB_TIMES_SKIPPING_ULEB
# CHECK-NEXT:   Imm:             0
# CHECK-NEXT:   ULEBExtraData:   [ 0x2, 0x8 ]
# CHECK-NEXT:   Symbol:          ''
# CHECK-NEXT:   Opcode:          BIND_OPCODE_SET_ADDEND_SLEB
# CHECK-NEXT:   Imm:             0
# CHECK-NEXT:   SLEBExtraData:   [ 1 ]
# CHECK-NEXT:   Symbol:          ''
# CHECK-NEXT:   Opcode:          BIND_OPCODE_DO_BIND_ADD_ADDR_ULEB
# CHECK-NEXT:   Imm:             0
# CHECK-NEXT:   ULEBExtraData:   [ 0x1008 ]
# CHECK-NEXT:   Symbol:          ''
# CHECK-NEXT:   Opcode:          BIND_OPCODE_SET_ADDEND_SLEB
# CHECK-NEXT:   Imm:             0
# CHECK-NEXT:   SLEBExtraData:   [ 0 ]
# CHECK-NEXT:   Symbol:          ''
# CHECK-NEXT:   Opcode:          BIND_OPCODE_DO_BIND
# CHECK-NEXT:   Imm:             0
# CHECK-NEXT:   Symbol:          ''
# CHECK-NEXT:   Opcode:          BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM
# CHECK-NEXT:   Imm:             0
# CHECK-NEXT:   Symbol:          _bar
# CHECK-NEXT:   Opcode:          BIND_OPCODE_SET_TYPE_IMM
# CHECK-NEXT:   Imm:             1
# CHECK-NEXT:   Symbol:          ''
# CHECK-NEXT:   Opcode:          BIND_OPCODE_ADD_ADDR_ULEB
# CHECK-NEXT:   Imm:             0
# CHECK-NEXT:   ULEBExtraData:   [ 0xFFFFFFFFFFFFEFD0 ]
# CHECK-NEXT:   Symbol:          ''
# CHECK-NEXT:   Opcode:          BIND_OPCODE_DO_BIND_ADD_ADDR_ULEB
# CHECK-NEXT:   Imm:             0
# CHECK-NEXT:   ULEBExtraData:   [ 0x8 ]
# CHECK-NEXT:   Symbol:          ''
# CHECK-NEXT:   Opcode:          BIND_OPCODE_DO_BIND_ADD_ADDR_ULEB
# CHECK-NEXT:   Imm:             0
# CHECK-NEXT:   ULEBExtraData:   [ 0x1008 ]
# CHECK-NEXT:   Symbol:          ''
# CHECK-NEXT:   Opcode:          BIND_OPCODE_DO_BIND
# CHECK-NEXT:   Imm:             0
# CHECK-NEXT:   Symbol:          ''
# CHECK-NEXT:   Opcode:          BIND_OPCODE_DONE
# CHECK-NEXT:   Imm:             0
# CHECK-NEXT:   Symbol:          ''

# RUN: llvm-objdump --macho --bind %t/test | FileCheck %s --check-prefix=BIND
# BIND:       Bind table:
# BIND-NEXT:  segment  section   address      type       addend dylib     symbol
# BIND-NEXT:  __DATA   __data    0x100001000  pointer         0 libfoo    _foo
# BIND-NEXT:  __DATA   __data    0x100001010  pointer         0 libfoo    _foo
# BIND-NEXT:  __DATA   __data    0x100001020  pointer         1 libfoo    _foo
# BIND-NEXT:  __DATA   __data    0x100002030  pointer         0 libfoo    _foo
# BIND-NEXT:  __DATA   __data    0x100001008  pointer         0 libfoo    _bar
# BIND-NEXT:  __DATA   __data    0x100001018  pointer         0 libfoo    _bar
# BIND-NEXT:  __DATA   __data    0x100002028  pointer         0 libfoo    _bar
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
.quad _foo+1
.zero 0x1000
.quad _bar
.quad _foo

.globl _main
.text
_main:
