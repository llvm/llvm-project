# REQUIRES: x86, aarch64
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin --defsym PTR64=0 %t/test.s -o %t/test.o
# RUN: %lld -O2 -dylib %t/foo.o -o %t/libfoo.dylib
# RUN: %lld -O2 -lSystem %t/test.o %t/libfoo.dylib -o %t/test-x86_64

## Test (64-bit):
## 1/ We emit exactly one BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM per symbol.
## 2/ Combine BIND_OPCODE_DO_BIND and BIND_OPCODE_ADD_ADDR_ULEB pairs.
## 3/ Compact BIND_OPCODE_DO_BIND_ADD_ADDR_ULEB
## 4/ Use BIND_OPCODE_DO_BIND_ADD_ADDR_IMM_SCALED if possible.
# RUN: obj2yaml %t/test-x86_64 | FileCheck %s

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
# CHECK-NEXT:   Opcode:          BIND_OPCODE_DO_BIND_ADD_ADDR_IMM_SCALED
# CHECK-NEXT:   Imm:             1
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

# RUN: llvm-mc -filetype=obj -triple=arm64_32-apple-darwin %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=arm64_32-apple-darwin --defsym PTR32=0 %t/test.s -o %t/test.o
# RUN: %lld-watchos -O2 -dylib %t/foo.o -o %t/libfoo.dylib
# RUN: %lld-watchos -O2 -dylib %t/test.o %t/libfoo.dylib -o %t/libtest-arm64_32.dylib

## Test (32-bit):
## 1/ We emit exactly one BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM per symbol.
## 2/ Combine BIND_OPCODE_DO_BIND and BIND_OPCODE_ADD_ADDR_ULEB pairs.
## 3/ Compact BIND_OPCODE_DO_BIND_ADD_ADDR_ULEB
## 4/ Use BIND_OPCODE_DO_BIND_ADD_ADDR_IMM_SCALED if possible.
# RUN: obj2yaml %t/libtest-arm64_32.dylib | FileCheck %s --check-prefix=CHECK32

# CHECK32:      BindOpcodes:
# CHECK32-NEXT:   Opcode:          BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM
# CHECK32-NEXT:   Imm:             0
# CHECK32-NEXT:   Symbol:          _foo
# CHECK32-NEXT:   Opcode:          BIND_OPCODE_SET_TYPE_IMM
# CHECK32-NEXT:   Imm:             1
# CHECK32-NEXT:   Symbol:          ''
# CHECK32-NEXT:   Opcode:          BIND_OPCODE_SET_DYLIB_ORDINAL_IMM
# CHECK32-NEXT:   Imm:             2
# CHECK32-NEXT:   Symbol:          ''
# CHECK32-NEXT:   Opcode:          BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB
# CHECK32-NEXT:   Imm:             1
# CHECK32-NEXT:   ULEBExtraData:   [ 0x0 ]
# CHECK32-NEXT:   Symbol:          ''
# CHECK32-NEXT:   Opcode:          BIND_OPCODE_DO_BIND_ULEB_TIMES_SKIPPING_ULEB
# CHECK32-NEXT:   Imm:             0
# CHECK32-NEXT:   ULEBExtraData:   [ 0x2, 0x4 ]
# CHECK32-NEXT:   Symbol:          ''
# CHECK32-NEXT:   Opcode:          BIND_OPCODE_SET_ADDEND_SLEB
# CHECK32-NEXT:   Imm:             0
# CHECK32-NEXT:   SLEBExtraData:   [ 1 ]
# CHECK32-NEXT:   Symbol:          ''
# CHECK32-NEXT:   Opcode:          BIND_OPCODE_DO_BIND_ADD_ADDR_ULEB
# CHECK32-NEXT:   Imm:             0
# CHECK32-NEXT:   ULEBExtraData:   [ 0x1004 ]
# CHECK32-NEXT:   Symbol:          ''
# CHECK32-NEXT:   Opcode:          BIND_OPCODE_SET_ADDEND_SLEB
# CHECK32-NEXT:   Imm:             0
# CHECK32-NEXT:   SLEBExtraData:   [ 0 ]
# CHECK32-NEXT:   Symbol:          ''
# CHECK32-NEXT:   Opcode:          BIND_OPCODE_DO_BIND
# CHECK32-NEXT:   Imm:             0
# CHECK32-NEXT:   Symbol:          ''
# CHECK32-NEXT:   Opcode:          BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM
# CHECK32-NEXT:   Imm:             0
# CHECK32-NEXT:   Symbol:          _bar
# CHECK32-NEXT:   Opcode:          BIND_OPCODE_SET_TYPE_IMM
# CHECK32-NEXT:   Imm:             1
# CHECK32-NEXT:   Symbol:          ''
# CHECK32-NEXT:   Opcode:          BIND_OPCODE_ADD_ADDR_ULEB
# CHECK32-NEXT:   Imm:             0
# CHECK32-NEXT:   ULEBExtraData:   [ 0xFFFFFFFFFFFFEFE8 ]
# CHECK32-NEXT:   Symbol:          ''
# CHECK32-NEXT:   Opcode:          BIND_OPCODE_DO_BIND_ADD_ADDR_IMM_SCALED
# CHECK32-NEXT:   Imm:             1
# CHECK32-NEXT:   Symbol:          ''
# CHECK32-NEXT:   Opcode:          BIND_OPCODE_DO_BIND_ADD_ADDR_ULEB
# CHECK32-NEXT:   Imm:             0
# CHECK32-NEXT:   ULEBExtraData:   [ 0x1004 ]
# CHECK32-NEXT:   Symbol:          ''
# CHECK32-NEXT:   Opcode:          BIND_OPCODE_DO_BIND
# CHECK32-NEXT:   Imm:             0
# CHECK32-NEXT:   Symbol:          ''
# CHECK32-NEXT:   Opcode:          BIND_OPCODE_DONE
# CHECK32-NEXT:   Imm:             0
# CHECK32-NEXT:   Symbol:          ''

# RUN: llvm-objdump --macho --bind %t/test-x86_64 | FileCheck %s -D#PTR=8 --check-prefix=BIND
# RUN: llvm-objdump --macho --bind %t/libtest-arm64_32.dylib | FileCheck %s -D#PTR=4 --check-prefix=BIND
# BIND:       Bind table:
# BIND-NEXT:  segment  section   address                               type       addend dylib     symbol
# BIND-NEXT:  __DATA   __data    0x[[#%X,DATA:]]                       pointer         0 libfoo    _foo
# BIND-NEXT:  __DATA   __data    0x[[#%.8X,DATA + mul(PTR, 2)]]        pointer         0 libfoo    _foo
# BIND-NEXT:  __DATA   __data    0x[[#%.8X,DATA + mul(PTR, 4)]]        pointer         1 libfoo    _foo
# BIND-NEXT:  __DATA   __data    0x[[#%.8X,DATA + 4096 + mul(PTR, 6)]] pointer         0 libfoo    _foo
# BIND-NEXT:  __DATA   __data    0x[[#%.8X,DATA + PTR]]                pointer         0 libfoo    _bar
# BIND-NEXT:  __DATA   __data    0x[[#%.8X,DATA + mul(PTR, 3)]]        pointer         0 libfoo    _bar
# BIND-NEXT:  __DATA   __data    0x[[#%.8X,DATA + 4096 + mul(PTR, 5)]] pointer         0 libfoo    _bar
# BIND-EMPTY:

#--- foo.s
.globl _foo, _bar
_foo:
  .space 4
_bar:
  .space 4

#--- test.s
.ifdef PTR64
.macro ptr val
  .quad \val
.endm
.endif

.ifdef PTR32
.macro ptr val
  .int \val
.endm
.endif

.data
ptr _foo
ptr _bar
ptr _foo
ptr _bar
ptr _foo+1
.zero 0x1000
ptr _bar
ptr _foo

.globl _main
.text
_main:
