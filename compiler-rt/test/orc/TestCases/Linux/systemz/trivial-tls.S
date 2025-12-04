// RUN: %clang -c -o %t %s
// RUN: %llvm_jitlink %t
//
// Test that basic ELF TLS work by adding together TLSs with values
// 0, 1, and -1, and returning the result (0 for success). This setup
// tests both zero-initialized (.tbss) and non-zero-initialized
// (.tdata) sections.

        .section        .data.rel.ro,"aw",@progbits
        .p2align        3, 0x0                          # -- Begin function main
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
main:                                   # @main
# %bb.0:                                # %entry
        stmg    %r11, %r15, 88(%r15)
        aghi    %r15, -192
        lgr     %r11, %r15
        mvhi    188(%r11), 0
        lgrl    %r2, .LCPI0_0 
        larl    %r12, _GLOBAL_OFFSET_TABLE_
        stg     %r12, 160(%r11)                 # 8-byte Spill
        brasl   %r14, __tls_get_offset@PLT:tls_gdcall:x
        lg      %r12, 160(%r11)                 # 8-byte Reload
        ear     %r1, %a0
                                        # implicit-def: $r0d
        lr      %r0, %r1
        sllg    %r1, %r0, 32
        ear     %r0, %a1
        lr      %r1, %r0
        stg     %r1, 176(%r11)                  # 8-byte Spill
        l       %r0, 0(%r2,%r1)
        st      %r0, 172(%r11)                  # 4-byte Spill
        lgrl    %r2, .LCPI0_1 
        brasl   %r14, __tls_get_offset@PLT:tls_gdcall:y
        lg      %r12, 160(%r11)                 # 8-byte Reload
        l       %r0, 172(%r11)                  # 4-byte Reload
        lg      %r1, 176(%r11)                  # 8-byte Reload
        l       %r1, 0(%r2,%r1)
        ar      %r0, %r1
        st      %r0, 184(%r11)                  # 4-byte Spill
        lgrl    %r2, .LCPI0_2 
        brasl   %r14, __tls_get_offset@PLT:tls_gdcall:z
        lg      %r1, 176(%r11)                  # 8-byte Reload
        l       %r0, 184(%r11)                  # 4-byte Reload
        l       %r1, 0(%r2,%r1)
        ar      %r0, %r1
        lgfr    %r2, %r0
        lmg     %r11, %r15, 280(%r11)
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
