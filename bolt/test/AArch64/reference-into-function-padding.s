## Check that padding after a function is not treated as part of the function.

# RUN: %clang %cflags %s -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --print-disasm 2>&1 | FileCheck %s

## foo is 12 bytes, but its MaxSize also covers the padding and the unnamed code.
# CHECK: Binary Function "foo" after disassembly
# CHECK: Size        : 0xc
# CHECK: MaxSize     : 0x20

## The references go to the old address, not to a label inside foo.
# CHECK: bl	{{.*}}SYMBOLat0x[[#%x,]]
# CHECK-NEXT: adrp	x1, {{.*}}SYMBOLat0x[[#%x,]]
# CHECK-NOT: .Ltmp
# CHECK: End of Function "foo"

        .text
        .globl  foo
        .type   foo,%function
foo:
        .reloc  ., R_AARCH64_CALL26, .text+16
        .inst   0x94000000              // bl  <unnamed>
        .reloc  ., R_AARCH64_ADR_PREL_PG_HI21, .text+16
        .inst   0x90000001              // adrp x1, <unnamed>
        ret
        .size   foo, .-foo
        .word   0                       // zero padding (udf #0)
        ret                             // foo+16: unnamed code, no symbol
        .balign 16
        .globl  next
        .type   next,%function
next:
        ret
        .size   next, .-next
        .globl  _start
_start:
        bl      foo
        bl      next
        mov     w0, #0
        ret
