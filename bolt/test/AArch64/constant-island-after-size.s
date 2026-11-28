## Check that a constant island located between a function's Size and MaxSize is
## still recognized as part of the function.

# RUN: %clang %cflags -no-pie %s -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --print-disasm --print-only=bar 2>&1 | \
# RUN:   FileCheck %s

# CHECK: Binary Function "bar" after disassembly
# CHECK: adrp	x1, ISLANDat0x{{[0-9a-f]+}}
# CHECK-NEXT: add	x1, x1, :lo12:ISLANDat0x{{[0-9a-f]+}}

        .text
        .balign 16
        .globl  foo
        .type   foo,%function
foo:
        mov     w0, #1
        nop
        nop
        ret
        .size   foo, .-foo
.Lpool:                                 // literal pool right after foo's .size
        .xword  foo
        .balign 16
        .globl  bar
        .type   bar,%function
bar:
        adrp    x1, .Lpool
        add     x1, x1, :lo12:.Lpool
        ldr     x2, [x1]                // pointer read from the pool
        adrp    x3, foo
        add     x3, x3, :lo12:foo
        mov     w0, #0
        cmp     x2, x3
        b.eq    1f
        mov     w0, #3
1:      ret
        .size   bar, .-bar
        .globl  _start
_start:
        bl      bar
        mov     w8, #93
        svc     #0
