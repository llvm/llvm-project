# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld --branch-to-branch %t.o -shared -o %t
# RUN: llvm-objdump -d --show-all-symbols %t | FileCheck %s
# RUN: ld.lld -O2 %t.o -shared -o %t
# RUN: llvm-objdump -d --show-all-symbols %t | FileCheck %s
# RUN: ld.lld %t.o -shared -o %t
# RUN: llvm-objdump -d --show-all-symbols %t | FileCheck --check-prefix=O0 %s

## Mostly positive cases, except for f2.
.section .text.jt1,"ax",@llvm_cfi_jump_table,8
## Function fits.
f1:
jmp f1.cfi
.balign 8, 0xcc

## Function too large.
f2:
jmp f2.cfi
.balign 8, 0xcc

## Function too large, but may be placed at the end.
## Because this causes the jump table to move, it is tested below.
f3:
jmp f3.cfi
.balign 8, 0xcc

## Mostly negative cases, except for f4.
.section .text.jt2,"ax",@llvm_cfi_jump_table,16
## Function already moved into jt1.
# CHECK:      <f1a>:
# CHECK-NEXT:   jmp {{.*}} <f1.cfi>
f1a:
jmp f1.cfi
.balign 16, 0xcc

## Function already moved into jt1.
# CHECK:      <f3a>:
# CHECK-NEXT:   jmp {{.*}} <f3.cfi>
f3a:
jmp f3.cfi
.balign 16, 0xcc

## Function too large for jt1 but small enough for jt2.
# CHECK:      <f4>:
# CHECK-NEXT: <f4.cfi>:
# CHECK-NEXT:   retq   $0x4
# O0:         <f4>:
# O0-NEXT:      jmp {{.*}} <f4.cfi>
f4:
jmp f4.cfi
.balign 16, 0xcc

## Function too large for jt2.
# CHECK:      <f5>:
# CHECK-NEXT:   jmp {{.*}} <f5.cfi>
f5:
jmp f5.cfi
.balign 16, 0xcc

## Branch target not at start of section.
# CHECK:      <f6>:
# CHECK-NEXT:   jmp {{.*}} <f6.cfi>
f6:
jmp f6.cfi
.balign 16, 0xcc

## Overaligned section.
# CHECK:      <f7>:
# CHECK-NEXT:   jmp {{.*}} <f7.cfi>
f7:
jmp f7.cfi
.balign 16, 0xcc

## Branch to IFUNC.
# CHECK:      <f8>:
# CHECK-NEXT:   jmp 0x[[IPLT:[0-9a-f]*]]
f8:
jmp f8.cfi
.balign 16, 0xcc

## Unexpected number of relocations in entry.
# CHECK:      <f9>:
# CHECK-NEXT:   jmp {{.*}} <f9.cfi>
# CHECK-NEXT: jmp {{.*}} <f9.cfi>
f9:
jmp f9.cfi
jmp f9.cfi
.balign 16, 0xcc

## Branch to different output section.
f10:
jmp f10.cfi
.balign 16, 0xcc

## Branch via PLT to STB_GLOBAL symbol.
# CHECK:      <f11>:
# CHECK-NEXT:   jmp {{.*}} <f11.cfi@plt>
f11:
jmp f11.cfi
.balign 16, 0xcc

## Invalid jumptable: entsize unset.
# CHECK:      <f12>:
# CHECK-NEXT:   jmp {{.*}} <f12.cfi>
.section .text.jt3,"ax",@0x6fff4c0e
f12:
jmp f12.cfi
.balign 8, 0xcc

## Invalid jumptable: size not a multiple of entsize.
# CHECK:      <f13>:
# CHECK-NEXT:   jmp {{.*}} <f13.cfi>
.section .text.jt4,"ax",@llvm_cfi_jump_table,8
f13:
jmp f13.cfi

## Jumptable alignment > entsize prevents it from being moved before last
## function, but moving non-last functions into the jumptable should work.
# CHECK:      <f14>:
# CHECK-NEXT: <f14.cfi>:
# CHECK-NEXT:   retq   $0xe
.section .text.jt5,"ax",@llvm_cfi_jump_table,8
.balign 16
f14:
jmp f14.cfi
.balign 8, 0xcc

## Empty target section.
# CHECK:      <f15>:
# CHECK-NEXT:   jmp {{.*}} <f15.cfi>
f15:
jmp f15.cfi
.balign 8, 0xcc

# CHECK:      <f16>:
# CHECK-NEXT:   jmp {{.*}} <f16.cfi>
f16:
jmp f16.cfi
.balign 8, 0xcc

## Invalid last jumptable entry: no relocations at all.

# CHECK:      <f17>:
# CHECK-NEXT:   retq $0x11
.section .text.jt6,"ax",@llvm_cfi_jump_table,16
f17:
ret $17
.balign 16, 0xcc

.section .text.f18,"ax",@progbits
f18.cfi:
ret $18

## Invalid last jumptable entry: no relocations covering entry.

# CHECK:      <f18>:
# CHECK-NEXT: <f18.cfi>:
# CHECK-NEXT:   retq   $0x12
.section .text.jt7,"ax",@llvm_cfi_jump_table,16
f18:
jmp f18.cfi
.balign 16, 0xcc

# CHECK:      <f19>:
# CHECK-NEXT:   retq $0x13
f19:
ret $19
.balign 16, 0xcc

.section .text.f20,"ax",@progbits
f20.cfi:
ret $20

## Invalid last jumptable entry: unexpected number of relocations.

# CHECK:      <f20>:
# CHECK-NEXT:   jmp {{.*}} <f20.cfi>
# CHECK-NEXT:   jmp {{.*}} <f20.cfi>
.section .text.jt8,"ax",@llvm_cfi_jump_table,16
f20:
jmp f20.cfi
jmp f20.cfi
.balign 16, 0xcc

## Similar case to f3, but without a padding section.
# CHECK:      <f21>:
# CHECK-NEXT: <f21.cfi>:
# CHECK-NEXT:   retq   $0x15
.section .text.jt9,"ax",@llvm_cfi_jump_table,8
.zero 4096
f21:
jmp f21.cfi
.balign 8, 0xcc

.section .text.f21,"ax",@progbits
.balign 4096
f21.cfi:
ret $21


# CHECK:      <f1>:
# CHECK-NEXT: <f1.cfi>:
# CHECK-NEXT:   retq   $0x1
.section .text.f1,"ax",@progbits
f1.cfi:
ret $1

# CHECK:      <f2>:
# CHECK-NEXT:   jmp {{.*}} <f2.cfi>
.section .text.f2,"ax",@progbits
f2.cfi:
ret $2
.zero 16

## Overalignment should trigger emitting enough padding behind the jump table to
## make these appear at the same address.
# CHECK:      <f3>:
# CHECK-NEXT: <f3.cfi>:
# CHECK-NEXT:   retq   $0x3
.section .text.f3,"ax",@progbits
.balign 64
f3.cfi:
ret $3
.zero 16

.section .text.f4,"ax",@progbits
f4.cfi:
ret $4
.zero 13

.section .text.f5,"ax",@progbits
f5.cfi:
ret $5
.zero 14

.section .text.f6,"ax",@progbits
nop
f6.cfi:
ret $6

.section .text.f7,"ax",@progbits
.balign 32
f7.cfi:
ret $7

.section .text.f8,"ax",@progbits
.type f8.cfi,@gnu_indirect_function
f8.cfi:
ret $8

.section .text.f9,"ax",@progbits
f9.cfi:
ret $9

.section foo,"ax",@progbits
f10.cfi:
ret $10

.section .text.f11,"ax",@progbits
.globl f11.cfi
f11.cfi:
ret $11

.section .text.f12,"ax",@progbits
f12.cfi:
ret $12

.section .text.f13,"ax",@progbits
f13.cfi:
ret $13

.section .text.f14,"ax",@progbits
f14.cfi:
ret $14

.section .text.f15,"ax",@progbits
f15.cfi:

.section .text.f16,"ax",@progbits
.balign 64
f16.cfi:
ret $16
.zero 16

## Test interaction between CFI jump table relaxation and section placement.

## Case 1: Jump table is in .sec_a and has entries in both .sec_a and .sec_b.
## The last entry is in .sec_b, but because an entry targets .sec_a, the jump
## table must not be moved into .sec_b.
.section .sec_a,"ax",@llvm_cfi_jump_table,8,unique,1
f22:
jmp f22.cfi
.balign 8, 0xcc
f23:
jmp f23.cfi
.balign 8, 0xcc

.section .sec_a,"ax",@progbits,unique,2
f22.cfi:
ret $22
.zero 16

.section .sec_b,"ax",@progbits,unique,3
f23.cfi:
ret $23

## Case 2: Jump table is in .sec_a, and its last entry is in .sec_a.
## Moving the jump table before the last entry keeps it in .sec_a, which
## is acceptable even if an earlier entry is in .sec_b.
.section .sec_a,"ax",@llvm_cfi_jump_table,8,unique,4
f24:
jmp f24.cfi
.balign 8, 0xcc
f25:
jmp f25.cfi
.balign 8, 0xcc

.section .sec_b,"ax",@progbits,unique,5
f24.cfi:
ret $24
.zero 16

.section .sec_a,"ax",@progbits,unique,6
f25.cfi:
ret $25

## Case 3: Jump table is in .sec_a, but ALL of its targets are in .sec_b.
## Since no entries target .sec_a, the jump table can be moved before the
## last entry into .sec_b.
.section .sec_a,"ax",@llvm_cfi_jump_table,8,unique,7
f26:
jmp f26.cfi
.balign 8, 0xcc
f27:
jmp f27.cfi
.balign 8, 0xcc

.section .sec_b,"ax",@progbits,unique,8
f26.cfi:
ret $26
.zero 16

.section .sec_b,"ax",@progbits,unique,9
f27.cfi:
ret $27

# CHECK: Disassembly of section .sec_a:

## Case 1: jt stays in .sec_a and both entries remain jumps.
# CHECK:      <f22>:
# CHECK-NEXT:   jmp {{.*}} <f22.cfi>
# CHECK:      <f23>:
# CHECK-NEXT:   jmp {{.*}} <f23.cfi>
# CHECK:      <f22.cfi>:
# CHECK-NEXT:   retq $0x16

## Case 2: jt stays in .sec_a, f24 remains a jump, and f25 is relaxed.
# CHECK:      <f24>:
# CHECK-NEXT:   jmp {{.*}} <f24.cfi>
# CHECK:      <f25>:
# CHECK-NEXT: <f25.cfi>:
# CHECK-NEXT:   retq $0x19
# O0:         <f25>:
# O0-NEXT:      jmp {{.*}} <f25.cfi>

# CHECK: Disassembly of section .sec_b:

## Case 3: jt is moved into .sec_b, f26 remains a jump, and f27 is relaxed.
# CHECK:      <f26>:
# CHECK-NEXT:   jmp {{.*}} <f26.cfi>
# CHECK:      <f27>:
# CHECK-NEXT: <f27.cfi>:
# CHECK-NEXT:   retq $0x1b
# O0:         <f27>:
# O0-NEXT:      jmp {{.*}} <f27.cfi>

# CHECK:      <.iplt>:
# CHECK-NEXT:   [[IPLT]]:
