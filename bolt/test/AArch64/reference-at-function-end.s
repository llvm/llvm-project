## An address at the end of a function cannot be told apart from
## the start of whatever follows it, so it is no longer attributed to foo and is
## kept as the original absolute address.

# RUN: %clang %cflags %s -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --print-disasm --print-only=foo 2>&1 | \
# RUN:   FileCheck %s

# CHECK: Binary Function "foo" after disassembly
# CHECK: Size        : 0x10
# CHECK: MaxSize     : 0x10
# CHECK-NOT: foo+
# CHECK: adrp	x1, SYMBOLat0x{{[0-9a-f]+}}
# CHECK-NEXT: add	x1, x1, :lo12:SYMBOLat0x{{[0-9a-f]+}}

        .text
        .balign 16
        .globl  _start
        .type   _start,%function
_start:
        bl      foo
        mov     w8, #93
        svc     #0
        .balign 16
        .globl  foo
        .type   foo,%function
foo:
        .reloc  ., R_AARCH64_ADR_PREL_PG_HI21, .text+32
        .inst   0x90000001              // adrp x1, <end of foo>
        .reloc  ., R_AARCH64_ADD_ABS_LO12_NC, .text+32
        .inst   0x91000021              // add x1, x1, :lo12:<end of foo>
        mov     w0, #0
        ret
        .size   foo, .-foo              // foo is the last function in .text
        .data
        .xword  1
