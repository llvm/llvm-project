# REQUIRES: x86
# RUN: llvm-mc -triple=x86_64-windows-msvc -filetype=obj -o %t.obj %s
# RUN: lld-link %t.obj /out:%t.exe /entry:main /subsystem:console
# RUN: llvm-objdump -s %t.exe | FileCheck %s

# CHECK: Contents of section .rdata:
# CHECK-NEXT:  140002000 68656c6c 6f20776f 726c6400 6fa26ca4  hello world.o.l.
# CHECK-NEXT:  140002010 0068656c 6c6f2077 6f726c64 00006865  .hello world..he
# CHECK-NEXT:  140002020 6c6c6f20 776f726c 64006800 65006c00  llo world.h.e.l.
# CHECK-NEXT:  140002030 6c006f00 20007700 6f007200 6c006400  l.o. .w.o.r.l.d.
# CHECK-NEXT:  140002040 0000                                 ..

# CHECK: Contents of section .text:
.globl main
main:
# CHECK-NEXT: 140003000 11200040 01000000 17200040 01000000
.8byte "??_C@_0M@LACCCNMM@hello?5world?$AA@"
.8byte "??_C@_05MCBCHHEJ@world?$AA@"
# CHECK-NEXT: 140003010 2a200040 01000000 36200040 01000000
.8byte "??_C@_1BI@HHJHKLLN@?$AAh?$AAe?$AAl?$AAl?$AAo?$AA?5?$AAw?$AAo?$AAr?$AAl?$AAd?$AA?$AA@"
.8byte "??_C@_1M@NBBDDHIO@?$AAw?$AAo?$AAr?$AAl?$AAd?$AA?$AA@"
# CHECK-NEXT: 140003020 00200040 01000000 0c200040 01000000
.8byte "??_D@not_a_string_literal"
.8byte "??_C@string_literal_with_relocs"
# CHECK-NEXT: 140003030 00100040 01000000 1e200040 01000000
.8byte "??_C@string_literal_in_wrong_section"
.8byte "??_C@overaligned_string_literal"

.section .rdata,"dr",discard,"??_C@_0M@LACCCNMM@hello?5world?$AA@"
.globl "??_C@_0M@LACCCNMM@hello?5world?$AA@"
"??_C@_0M@LACCCNMM@hello?5world?$AA@":
.asciz "hello world"

.section .rdata,"dr",discard,"??_C@_05MCBCHHEJ@world?$AA@"
.globl "??_C@_05MCBCHHEJ@world?$AA@"
"??_C@_05MCBCHHEJ@world?$AA@":
.asciz "world"

.section .rdata,"dr",discard,"??_C@_1BI@HHJHKLLN@?$AAh?$AAe?$AAl?$AAl?$AAo?$AA?5?$AAw?$AAo?$AAr?$AAl?$AAd?$AA?$AA@"
.globl "??_C@_1BI@HHJHKLLN@?$AAh?$AAe?$AAl?$AAl?$AAo?$AA?5?$AAw?$AAo?$AAr?$AAl?$AAd?$AA?$AA@"
.p2align 1
"??_C@_1BI@HHJHKLLN@?$AAh?$AAe?$AAl?$AAl?$AAo?$AA?5?$AAw?$AAo?$AAr?$AAl?$AAd?$AA?$AA@":
.short 104
.short 101
.short 108
.short 108
.short 111
.short 32
.short 119
.short 111
.short 114
.short 108
.short 100
.short 0

.section .rdata,"dr",discard,"??_C@_1M@NBBDDHIO@?$AAw?$AAo?$AAr?$AAl?$AAd?$AA?$AA@"
.globl "??_C@_1M@NBBDDHIO@?$AAw?$AAo?$AAr?$AAl?$AAd?$AA?$AA@"
.p2align 1
"??_C@_1M@NBBDDHIO@?$AAw?$AAo?$AAr?$AAl?$AAd?$AA?$AA@":
.short 119
.short 111
.short 114
.short 108
.short 100
.short 0

.section .data,"drw",discard,"??_C@string_literal_in_wrong_section"
.globl "??_C@string_literal_in_wrong_section"
"??_C@string_literal_in_wrong_section":
.asciz "hello world"

.section .rdata,"dr",discard,"??_D@not_a_string_literal"
.globl "??_D@not_a_string_literal"
"??_D@not_a_string_literal":
.asciz "hello world"

.section .rdata,"dr",discard,"??_C@string_literal_with_relocs"
.globl "??_C@string_literal_with_relocs"
"??_C@string_literal_with_relocs":
.4byte main + 111 + (114 << 8) + (108 << 16) + (100 << 24) # main + "orld"
.byte 0

.section .rdata,"dr",discard,"??_C@overaligned_string_literal"
.globl "??_C@overaligned_string_literal"
.p2align 1
"??_C@overaligned_string_literal":
.asciz "hello world"
