# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: %lld -dylib %t.o -o %t.dylib
# RUN: obj2yaml %t.dylib | FileCheck %s

.text
.globl _foo
_foo:

.data
# CHECK: RebaseOpcodes:
# CHECK-NEXT: Opcode:          REBASE_OPCODE_SET_TYPE_IMM
# CHECK-NEXT: Imm:             1
# CHECK-NEXT: Opcode:          REBASE_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB
# CHECK-NEXT: Imm:             1
# CHECK-NEXT: ExtraData:       [ 0x0 ]

## 1/ Single rebases with a gap after them are encoded as REBASE_OPCODE_DO_REBASE_ADD_ADDR_ULEB.
.quad _foo
.space 16
# CHECK-NEXT: Opcode:          REBASE_OPCODE_DO_REBASE_ADD_ADDR_ULEB
# CHECK-NEXT: Imm:             0
# CHECK-NEXT: ExtraData:       [ 0x10 ]

## 2/ Consecutive rebases are encoded as REBASE_OPCODE_DO_REBASE_IMM_TIMES.
.quad _foo
.quad _foo
.quad _foo
# CHECK-NEXT: Opcode:          REBASE_OPCODE_DO_REBASE_IMM_TIMES
# CHECK-NEXT: Imm:             3

## 3/ Gaps smaller than 16 words are encoded as REBASE_OPCODE_ADD_ADDR_IMM_SCALED.
.space 120
# CHECK-NEXT: Opcode:          REBASE_OPCODE_ADD_ADDR_IMM_SCALED
# CHECK-NEXT: Imm:             15

## 4/ Rebases with equal gaps betwen them are encoded as REBASE_OPCODE_DO_REBASE_ULEB_TIMES_SKIPPING_ULEB.
.quad _foo
.space 16
.quad _foo
.space 16
# CHECK-NEXT: Opcode:          REBASE_OPCODE_DO_REBASE_ULEB_TIMES_SKIPPING_ULEB
# CHECK-NEXT: Imm:             0
# CHECK-NEXT: ExtraData:       [ 0x2, 0x10 ]

## 5/ Rebase does not become a part of DO_REBASE_ULEB_TIMES_SKIPPING_ULEB if the next rebase is closer than the gap.
.quad _foo
.space 8
# CHECK-NEXT: Opcode:          REBASE_OPCODE_DO_REBASE_ADD_ADDR_ULEB
# CHECK-NEXT: Imm:             0
# CHECK-NEXT: ExtraData:       [ 0x8 ]

.quad _foo
.quad _foo
# CHECK-NEXT: Opcode:          REBASE_OPCODE_DO_REBASE_IMM_TIMES
# CHECK-NEXT: Imm:             2

## 6/ Large gaps are encoded as REBASE_OPCODE_ADD_ADDR_ULEB.
.space 128
# CHECK-NEXT: Opcode:          REBASE_OPCODE_ADD_ADDR_ULEB
# CHECK-NEXT: Imm:             0
# CHECK-NEXT: ExtraData:       [ 0x80 ]

.quad _foo
.space 8
.quad _foo
.space 8
.quad _foo
# CHECK-NEXT: Opcode:          REBASE_OPCODE_DO_REBASE_ULEB_TIMES_SKIPPING_ULEB
# CHECK-NEXT: Imm:             0
# CHECK-NEXT: ExtraData:       [ 0x3, 0x8 ]


## 7/ An add opcode is emitted if the next relocation is farther away than the DO_REBASE_ULEB_TIMES_SKIPPING_ULEB gap.
.space 16
.quad _foo
# CHECK-NEXT: Opcode:          REBASE_OPCODE_ADD_ADDR_IMM_SCALED
# CHECK-NEXT: Imm:             1
# CHECK-NEXT: Opcode:          REBASE_OPCODE_DO_REBASE_IMM_TIMES
# CHECK-NEXT: Imm:             1

## 8/ The rebase section is terminated by REBASE_OPCODE_DONE.
# CHECK-NEXT: Opcode:          REBASE_OPCODE_DONE
# CHECK-NEXT: Imm:             0
