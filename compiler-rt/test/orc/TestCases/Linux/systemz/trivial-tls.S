// RUN: %clang -c -o %t %s
// RUN: %llvm_jitlink %t
//
// Test that basic ELF TLS work by adding together TLSs with values
// 0, 1, and -1, and returning the result (0 for success). This setup
// tests both zero-initialized (.tbss) and non-zero-initialized
// (.tdata) sections.

        .section        .data.rel.ro,"aw",@progbits
        .p2align        3, 0x0
.LCPI0_0:
        .quad   x@TLSGD
.LCPI0_1:
        .quad   y@TLSGD
.LCPI0_2:
        .quad   z@TLSGD

        .text
        .globl  main
        .p2align        4
        .type   main,@function
main:
        stmg    %r10, %r15, 80(%r15)
        aghi    %r15, -160
        lgrl    %r2, .LCPI0_0
        larl    %r12, _GLOBAL_OFFSET_TABLE_
        brasl   %r14, __tls_get_offset@PLT:tls_gdcall:x
        lgr     %r13, %r2
        lgrl    %r2, .LCPI0_1
        brasl   %r14, __tls_get_offset@PLT:tls_gdcall:y
        ear     %r0, %a0
        sllg    %r11, %r0, 32
        ear     %r11, %a1
        l       %r10, 0(%r2,%r11)
        lgrl    %r2, .LCPI0_2
        a       %r10, 0(%r13,%r11)
        brasl   %r14, __tls_get_offset@PLT:tls_gdcall:z
        a       %r10, 0(%r2,%r11)
        lgfr    %r2, %r10
        lmg     %r10, %r15, 240(%r15)
        br      %r14
.Lfunc_end0:
        .size   main, .Lfunc_end0-main


        .type   x,@object                       # @x
        .section        .tbss,"awT",@nobits
        .globl  x
        .p2align        2, 0x0
x:
        .long   0                               # 0x0
        .size   x, 4

        .type   y,@object                       # @y
        .section        .tdata,"awT",@progbits
        .globl  y
        .p2align        2, 0x0
y:
        .long   1                               # 0x1
        .size   y, 4

        .type   z,@object                       # @z
        .globl  z
        .p2align        2, 0x0
z:
        .long   4294967295                      # 0xffffffff
        .size   z, 4
