## Check that rewriting an AUIPC/JALR call pair preserves an input offset used
## as a DWARF lexical-scope boundary. The first call grows from eight to twelve
## bytes. The parent scope ends at the AUIPC of the second call, while its child
## ends at the intervening compressed NOP. Losing the AUIPC offset maps the
## parent's high_pc inside the preceding rewritten call and makes the child
## extend beyond its parent.

# REQUIRES: system-linux

# RUN: llvm-mc -triple riscv64 -mattr=+c -filetype obj -o %t.o %s
# RUN: ld.lld --no-relax --emit-relocs -e foo -o %t %t.o
# RUN: llvm-bolt --update-debug-sections -o %t.bolt %t
# RUN: llvm-dwarfdump --verify %t.bolt
# RUN: llvm-objdump -d --no-show-raw-insn %t.bolt > %t.out
# RUN: llvm-dwarfdump --debug-info %t.bolt >> %t.out
# RUN: FileCheck %s < %t.out

# CHECK-LABEL: <foo>:
# CHECK:      nop
# CHECK-NEXT: auipc
# CHECK-NEXT: jalr
# CHECK-NEXT: nop
# CHECK-NEXT: [[PARENT_END:[0-9a-f]+]]:{{.*}}nop
# CHECK-NEXT: auipc
# CHECK-NEXT: jalr
# CHECK:      DW_TAG_lexical_block
# CHECK:      DW_AT_low_pc
# CHECK-NEXT: DW_AT_high_pc {{.*}}0x{{0*}}[[PARENT_END]])

        .text
        .option norvc
        .option norelax
        .globl  foo
        .p2align 2
        .type   foo,@function
foo:
.Lfoo_begin:
        call    callee
        .option rvc
.Lchild_end:
        c.nop
        .option norvc
.Lparent_end:
        call    callee
        ret
.Lfoo_end:
        .size   foo, .-foo

        .skip   (1 << 21)

        .globl  callee
        .p2align 2
        .type   callee,@function
callee:
        ret
.Lcallee_end:
        .size   callee, .-callee

        .section .debug_abbrev,"",@progbits
        .byte   1                       # Abbrev code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   37                      # DW_AT_producer
        .byte   8                       # DW_FORM_string
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
        .byte   18                      # DW_AT_high_pc
        .byte   6                       # DW_FORM_data4
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   16                      # DW_AT_stmt_list
        .byte   23                      # DW_FORM_sec_offset
        .byte   0
        .byte   0
        .byte   2                       # Abbrev code
        .byte   46                      # DW_TAG_subprogram
        .byte   1                       # DW_CHILDREN_yes
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
        .byte   18                      # DW_AT_high_pc
        .byte   6                       # DW_FORM_data4
        .byte   0
        .byte   0
        .byte   3                       # Abbrev code
        .byte   11                      # DW_TAG_lexical_block
        .byte   1                       # DW_CHILDREN_yes
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
        .byte   18                      # DW_AT_high_pc
        .byte   6                       # DW_FORM_data4
        .byte   0
        .byte   0
        .byte   4                       # Abbrev code
        .byte   11                      # DW_TAG_lexical_block
        .byte   0                       # DW_CHILDREN_no
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
        .byte   18                      # DW_AT_high_pc
        .byte   6                       # DW_FORM_data4
        .byte   0
        .byte   0
        .byte   0

        .section .debug_info,"",@progbits
.Lcu_begin:
        .long   .Lcu_end-.Lcu_version
.Lcu_version:
        .short  4                       # DWARF version
        .long   .debug_abbrev
        .byte   8                       # Address size
        .byte   1                       # DW_TAG_compile_unit
        .asciz  "test producer"
        .quad   .Lfoo_begin
        .long   .Lcallee_end-.Lfoo_begin
        .asciz  "dwarf-scope-call-pair.s"
        .long   .Lline_table_start
        .byte   2                       # DW_TAG_subprogram
        .asciz  "foo"
        .quad   .Lfoo_begin
        .long   .Lfoo_end-.Lfoo_begin
        .byte   3                       # Parent lexical block
        .quad   .Lfoo_begin
        .long   .Lparent_end-.Lfoo_begin
        .byte   4                       # Child lexical block
        .quad   .Lfoo_begin
        .long   .Lchild_end-.Lfoo_begin
        .byte   0                       # End parent children
        .byte   0                       # End subprogram children
        .byte   0                       # End CU children
.Lcu_end:

        .section .debug_line,"",@progbits
.Lline_table_start:
        .long   .Lline_table_end-.Lline_version
.Lline_version:
        .short  4                       # DWARF version
        .long   .Lline_prologue_end-.Lline_prologue_start
.Lline_prologue_start:
        .byte   1                       # Minimum instruction length
        .byte   1                       # Maximum operations per instruction
        .byte   1                       # Default is_stmt
        .byte   -5                      # Line base
        .byte   14                      # Line range
        .byte   13                      # Opcode base
        .byte   0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1
        .byte   0                       # Include directory terminator
        .asciz  "dwarf-scope-call-pair.s"
        .uleb128 0                      # Directory index
        .uleb128 0                      # Modification time
        .uleb128 0                      # File size
        .byte   0                       # File table terminator
.Lline_prologue_end:
.Lline_table_end:

        .section ".note.GNU-stack","",@progbits
