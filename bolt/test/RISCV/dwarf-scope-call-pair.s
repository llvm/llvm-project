## A shrinking call changes the offsets of later scope boundaries within the
## same basic block. Both the AUIPC and JALR of a rewritten pair must map to the
## surviving call. Otherwise the low_pc of the first parent follows its child's
## low_pc, and the high_pc of the second child exceeds its parent's high_pc.
## Check exact endpoints too: verification alone misses misplaced boundaries
## when replacement NOPs are retained or the first call does not shrink.

# REQUIRES: system-linux

# RUN: llvm-mc -triple riscv64 -dwarf-version=4 -filetype=obj %s -o %t.o
# RUN: ld.lld --no-relax --emit-relocs --section-start=.text=0x10000 \
# RUN:   --section-start=.far=0x400000 -e foo %t.o -o %t
# RUN: llvm-dwarfdump --verify %t
# RUN: llvm-bolt %t --update-debug-sections --skip-funcs=far_callee -o %t.bolt
# RUN: llvm-dwarfdump --verify %t.bolt
# RUN: llvm-objdump -d --no-show-raw-insn %t.bolt > %t.out
# RUN: llvm-dwarfdump --debug-info --debug-line %t.bolt >> %t.out
# RUN: FileCheck %s --check-prefixes=CHECK,SHORT < %t.out

## BAT reverse lookups use the last entry at a shared output address. Keep the
## original JALR mapping after the AUIPC alias so branch profiles are unchanged.
# RUN: llvm-bolt %t --update-debug-sections --skip-funcs=far_callee \
# RUN:   --enable-bat -o %t.bat
# RUN: llvm-bat-dump %t.bat --dump-all | FileCheck %s --check-prefix=BAT

# BAT: BB mappings:
# BAT-NEXT: 0x0 -> 0x0 hash:
# BAT-NEXT: 0x4 -> 0x8 (branch)
# BAT-NEXT: 0x4 -> 0xc (branch)
# BAT-NEXT: 0xc -> 0x10 (branch)
# BAT-NEXT: 0x10 -> 0x14 (branch)
# BAT-NEXT: 0x10 -> 0x18 (branch)
# BAT-NEXT: 0x18 -> 0x1c (branch)
# BAT-NEXT: NumBlocks: 1

# RUN: llvm-bolt %t --update-debug-sections --skip-funcs=far_callee \
# RUN:   --keep-nops -o %t.keep
# RUN: llvm-dwarfdump --verify %t.keep
# RUN: llvm-objdump -d --no-show-raw-insn %t.keep > %t.keep.out
# RUN: llvm-dwarfdump --debug-info --debug-line %t.keep >> %t.keep.out
# RUN: FileCheck %s --check-prefixes=CHECK,SHORT,KEEP < %t.keep.out

# RUN: llvm-mc -triple riscv64 -dwarf-version=4 -filetype=obj \
# RUN:   --defsym FIRST_FAR=1 %s -o %t.far.o
# RUN: ld.lld --no-relax --emit-relocs --section-start=.text=0x10000 \
# RUN:   --section-start=.far=0x400000 -e foo %t.far.o -o %t.far
# RUN: llvm-bolt %t.far --update-debug-sections --skip-funcs=far_callee \
# RUN:   -o %t.far.bolt
# RUN: llvm-dwarfdump --verify %t.far.bolt
# RUN: llvm-objdump -d --no-show-raw-insn %t.far.bolt > %t.far.out
# RUN: llvm-dwarfdump --debug-info --debug-line %t.far.bolt >> %t.far.out
# RUN: FileCheck %s --check-prefixes=CHECK,LONG < %t.far.out

## An internal call makes BOLT preserve NOPs for this function even without
## --keep-nops. The AUIPC must not acquire two different output mappings.
# RUN: llvm-mc -triple riscv64 -dwarf-version=4 -filetype=obj \
# RUN:   --defsym INTERNAL=1 %s -o %t.internal.o
# RUN: ld.lld --no-relax --emit-relocs --section-start=.text=0x10000 \
# RUN:   --section-start=.far=0x400000 -e foo %t.internal.o -o %t.internal
# RUN: llvm-bolt %t.internal --update-debug-sections --skip-funcs=far_callee \
# RUN:   -o %t.internal.bolt
# RUN: llvm-dwarfdump --verify %t.internal.bolt
# RUN: llvm-objdump -d --no-show-raw-insn %t.internal.bolt > %t.internal.out
# RUN: llvm-dwarfdump --debug-info --debug-line %t.internal.bolt >> %t.internal.out
# RUN: FileCheck %s --check-prefixes=CHECK,SHORT,KEEP,INTERNAL < %t.internal.out

# CHECK-LABEL: <foo>:
# INTERNAL-NEXT: jal
# KEEP-NEXT: nop
# SHORT-NEXT: [[FIRST:[0-9a-f]+]]:{{.*}}jal {{.*}} <near_callee>
# LONG-NEXT: [[FIRST:[0-9a-f]+]]:{{.*}}auipc
# LONG-NEXT: jalr
# KEEP-NEXT: nop
# CHECK-NOT: nop
# CHECK: [[BEGIN:[0-9a-f]+]]:{{.*}}auipc
# CHECK-NEXT: jalr
# CHECK-NEXT: [[END:[0-9a-f]+]]:{{.*}}addi
# KEEP-NEXT: nop
# CHECK-NEXT: [[BOUNDARY:[0-9a-f]+]]:{{.*}}auipc
# CHECK-NEXT: jalr
# CHECK-NEXT: ret
# CHECK: DW_AT_name ("parent_high")
# CHECK-NEXT: DW_AT_low_pc
# CHECK-NEXT: DW_AT_high_pc (0x{{0*}}[[BOUNDARY]])
# CHECK: DW_AT_name ("child_high")
# CHECK-NEXT: DW_AT_low_pc
# CHECK-NEXT: DW_AT_high_pc (0x{{0*}}[[BOUNDARY]])
# CHECK: DW_AT_name ("parent_low")
# CHECK-NEXT: DW_AT_low_pc (0x{{0*}}[[BEGIN]])
# CHECK-NEXT: DW_AT_high_pc (0x{{0*}}[[END]])
# CHECK: DW_AT_name ("child_low")
# CHECK-NEXT: DW_AT_low_pc (0x{{0*}}[[BEGIN]])
# CHECK-NEXT: DW_AT_high_pc (0x{{0*}}[[END]])
# CHECK: 0x{{0*}}[[FIRST]] 10
# CHECK: 0x{{0*}}[[BEGIN]] 21
# CHECK: 0x{{0*}}[[END]] 30
# CHECK: 0x{{0*}}[[BOUNDARY]] 41

        .text
        .option norvc
        .option norelax
        .file 1 "dwarf-scope-call-pair.s"
        .globl foo
        .type foo,@function
foo:
        .ifdef INTERNAL
        jal ra, .Lfirst_call
        .endif
.Lfirst_call:
        .loc 1 10
        .ifdef FIRST_FAR
        call far_callee
        .else
        call near_callee
        .endif
.Lparent_begin:
        .loc 1 20
        .reloc ., R_RISCV_CALL_PLT, far_callee
        auipc ra, 0
.Lchild_begin:
        .loc 1 21
        jalr ra
.Lparent_end:
        .loc 1 30
        addi a0, a0, 1
.Lboundary_auipc:
        .loc 1 40
        .reloc ., R_RISCV_CALL_PLT, far_callee
        auipc ra, 0
.Lboundary_jalr:
        .loc 1 41
        jalr ra
        .loc 1 50
        ret
.Lfoo_end:
        .size foo, .-foo

        .globl near_callee
        .type near_callee,@function
near_callee:
        ret
        .size near_callee, .-near_callee

        .section .far,"ax",@progbits
        .globl far_callee
        .type far_callee,@function
far_callee:
        ret
        .size far_callee, .-far_callee

        .section .debug_abbrev,"",@progbits
        .byte 1, 0x11, 1
        .byte 0x03, 0x08
        .byte 0x11, 0x01
        .byte 0x12, 0x06
        .byte 0x10, 0x17
        .byte 0, 0

        .byte 2, 0x2e, 1
        .byte 0x03, 0x08
        .byte 0x11, 0x01
        .byte 0x12, 0x06
        .byte 0, 0

        .byte 3, 0x0b, 1
        .byte 0x03, 0x08
        .byte 0x11, 0x01
        .byte 0x12, 0x06
        .byte 0, 0

        .byte 4, 0x0b, 0
        .byte 0x03, 0x08
        .byte 0x11, 0x01
        .byte 0x12, 0x06
        .byte 0, 0
        .byte 0

        .section .debug_info,"",@progbits
        .long .Lcu_end-.Lcu_version
.Lcu_version:
        .short 4
        .long .debug_abbrev
        .byte 8
        .byte 1
        .asciz "dwarf-scope-call-pair.s"
        .quad foo
        .long .Lfoo_end-foo
        .long 0
        .byte 2
        .asciz "foo"
        .quad foo
        .long .Lfoo_end-foo
        .byte 3
        .asciz "parent_high"
        .quad foo
        .long .Lboundary_jalr-foo
        .byte 3
        .asciz "child_high"
        .quad foo
        .long .Lboundary_auipc-foo
        .byte 3
        .asciz "parent_low"
        .quad .Lparent_begin
        .long .Lparent_end-.Lparent_begin
        .byte 4
        .asciz "child_low"
        .quad .Lchild_begin
        .long .Lparent_end-.Lchild_begin
        .byte 0, 0, 0, 0, 0
.Lcu_end:

        .section .note.GNU-stack,"",@progbits
