.data
        .reloc ., R_AARCH64_ABS64, abs64
        .xword 0x1234567887654321

        .reloc ., R_AARCH64_ABS32, abs32
        .word 0x12344321

        .reloc ., R_AARCH64_ABS16, abs16
        .hword 0x1234

        .balign 16

        .reloc ., R_AARCH64_PREL64, data
        .xword 0x1234567887654321

        .reloc ., R_AARCH64_PREL32, data
        .word 0x12344321

        .reloc ., R_AARCH64_PREL16, data
        .hword 0x1234

.text
.globl _start
_start:
        // Full set of 4 instructions loading the constant 'abs64' and
        // adding 0x1234 to it.
        .reloc ., R_AARCH64_MOVW_UABS_G0_NC, abs64
        movz x0, #0x1234
        .reloc ., R_AARCH64_MOVW_UABS_G1_NC, abs64
        movk x0, #0x1234, lsl #16
        .reloc ., R_AARCH64_MOVW_UABS_G2_NC, abs64
        movk x0, #0x1234, lsl #32
        .reloc ., R_AARCH64_MOVW_UABS_G3,    abs64
        movk x0, #0x1234, lsl #48

        // The same, but this constant has ffff in the middle 32 bits,
        // forcing carries to be propagated.
        .reloc ., R_AARCH64_MOVW_UABS_G0_NC, big64
        movz x0, #0x1234
        .reloc ., R_AARCH64_MOVW_UABS_G1_NC, big64
        movk x0, #0x1234, lsl #16
        .reloc ., R_AARCH64_MOVW_UABS_G2_NC, big64
        movk x0, #0x1234, lsl #32
        .reloc ., R_AARCH64_MOVW_UABS_G3,    big64
        movk x0, #0x1234, lsl #48

        // Demonstrate that offsets are treated as signed: this one is
        // taken to be -0x1234. (If it were +0xedcc then you'd be able
        // to tell the difference by the carry into the second halfword.)
        .reloc ., R_AARCH64_MOVW_UABS_G0_NC, abs64
        movz x0, #0xedcc
        .reloc ., R_AARCH64_MOVW_UABS_G1_NC, abs64
        movk x0, #0xedcc, lsl #16
        .reloc ., R_AARCH64_MOVW_UABS_G2_NC, abs64
        movk x0, #0xedcc, lsl #32
        .reloc ., R_AARCH64_MOVW_UABS_G3,    abs64
        movk x0, #0xedcc, lsl #48

        // Check various bits of the ADR immediate, including in
        // particular the low 2 bits, which are not contiguous with the
        // rest in the encoding.
        .reloc ., R_AARCH64_ADR_PREL_LO21, pcrel
        adr x0, .+1
        .reloc ., R_AARCH64_ADR_PREL_LO21, pcrel
        adr x0, .+2
        .reloc ., R_AARCH64_ADR_PREL_LO21, pcrel
        adr x0, .+4
        .reloc ., R_AARCH64_ADR_PREL_LO21, pcrel
        adr x0, .+8
        .reloc ., R_AARCH64_ADR_PREL_LO21, pcrel
        adr x0, .+1<<19
        .reloc ., R_AARCH64_ADR_PREL_LO21, pcrel
        adr x0, .-1<<20

        // Now do the same with ADRP+ADD. But because the real ADRP
        // instruction shifts its immediate, we must account for that.
        .reloc ., R_AARCH64_ADR_PREL_PG_HI21, pcrel
        adrp x0, 1<<12
        .reloc ., R_AARCH64_ADD_ABS_LO12_NC,  pcrel
        add x0, x0, #1
        .reloc ., R_AARCH64_ADR_PREL_PG_HI21, pcrel
        adrp x0, 2<<12
        .reloc ., R_AARCH64_ADD_ABS_LO12_NC,  pcrel
        add x0, x0, #2
        .reloc ., R_AARCH64_ADR_PREL_PG_HI21, pcrel
        adrp x0, 4<<12
        .reloc ., R_AARCH64_ADD_ABS_LO12_NC,  pcrel
        add x0, x0, #4
        .reloc ., R_AARCH64_ADR_PREL_PG_HI21, pcrel
        adrp x0, 8<<12
        .reloc ., R_AARCH64_ADD_ABS_LO12_NC,  pcrel
        add x0, x0, #8
        // These high bits won't fit in the ADD immediate, so that becomes 0
        .reloc ., R_AARCH64_ADR_PREL_PG_HI21, pcrel
        adrp x0, 1<<(19+12)
        .reloc ., R_AARCH64_ADD_ABS_LO12_NC,  pcrel
        add x0, x0, #0
        .reloc ., R_AARCH64_ADR_PREL_PG_HI21, pcrel
        adrp x0, -1<<(20+12)
        .reloc ., R_AARCH64_ADD_ABS_LO12_NC,  pcrel
        add x0, x0, #0

        // Finally, an example with a full 21-bit addend
        .reloc ., R_AARCH64_ADR_PREL_PG_HI21, pcrel
        adrp x0, (0xfedcb-0x100000)<<12
        .reloc ., R_AARCH64_ADD_ABS_LO12_NC,  pcrel
        add x0, x0, #0xdcb

        // PC-relative loads, in which the 19-bit offset is shifted.
        // The input syntax is confusing here; I'd normally expect to
        // write ldr x0, [pc, #offset], but LLVM writes just #offset.
        .reloc ., R_AARCH64_LD_PREL_LO19,     pcrel
        ldr w0, #4
        .reloc ., R_AARCH64_LD_PREL_LO19,     pcrel
        ldr w0, #8
        .reloc ., R_AARCH64_LD_PREL_LO19,     pcrel
        ldr w0, #1<<19
        .reloc ., R_AARCH64_LD_PREL_LO19,     pcrel
        ldr w0, #-1<<20

        .reloc ., R_AARCH64_TSTBR14, branchtarget
        tbnz x1, #63, #4
        .reloc ., R_AARCH64_TSTBR14, branchtarget
        tbnz x1, #62, #8
        .reloc ., R_AARCH64_TSTBR14, branchtarget
        tbnz x1, #61, #1<<14
        .reloc ., R_AARCH64_TSTBR14, branchtarget
        tbnz x1, #60, #-1<<15

        // CONDBR19 is used for both cbz/cbnz and B.cond
        .reloc ., R_AARCH64_CONDBR19, branchtarget
        cbnz x2, #4
        .reloc ., R_AARCH64_CONDBR19, branchtarget
        b.eq #8
        .reloc ., R_AARCH64_CONDBR19, branchtarget
        cbz x2, #1<<19
        .reloc ., R_AARCH64_CONDBR19, branchtarget
        b.vs #-1<<20

        .reloc ., R_AARCH64_CALL26, calltarget
        bl #4
        .reloc ., R_AARCH64_CALL26, calltarget
        bl #8
        .reloc ., R_AARCH64_CALL26, calltarget
        bl #1<<24
        .reloc ., R_AARCH64_CALL26, calltarget
        bl #-1<<25

        .reloc ., R_AARCH64_JUMP26, calltarget
        b #4
        .reloc ., R_AARCH64_JUMP26, calltarget
        b #8
        .reloc ., R_AARCH64_JUMP26, calltarget
        b #1<<24
        .reloc ., R_AARCH64_JUMP26, calltarget
        b #-1<<25
