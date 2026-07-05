// REQUIRES: riscv
// RUN: rm -rf %t && split-file %s %t && cd %t

// RUN: llvm-mc -filetype=obj -triple=riscv64 a.s -o a.o
// RUN: ld.lld a.o -T eh-frame-non-zero-offset.t -o non-zero
// RUN: llvm-readelf --program-headers --unwind --symbols -x .eh_frame non-zero | FileCheck --check-prefix=NONZERO %s
// RUN: ld.lld a.o -T eh-frame-zero-offset.t -o zero
// RUN: llvm-readelf --program-headers --unwind --symbols -x .eh_frame zero | FileCheck --check-prefix=ZERO %s

// NONZERO:      {{[0-9]+}}: 0000000000000088 {{.*}} __eh_frame_start
// NONZERO-NEXT: {{[0-9]+}}: 00000000000000b4 {{.*}} __eh_frame_end

// NONZERO:      0x00000088 10000000 00000000 017a5200 01780101 .
// NONZERO-NEXT: 0x00000098 1b0c0200 10000000 18000000 5cffffff .
// NONZERO-NEXT: 0x000000a8 04000000 00000000 00000000          .

// ZERO:      {{[0-9]+}}: 0000000000000008 {{.*}} __eh_frame_start
// ZERO-NEXT: {{[0-9]+}}: 0000000000000034 {{.*}} __eh_frame_end

// ZERO:      0x00000008 10000000 00000000 017a5200 01780101 .
// ZERO-NEXT: 0x00000018 1b0c0200 10000000 18000000 dcffffff .
// ZERO-NEXT: 0x00000028 04000000 00000000 00000000          .

//--- eh-frame-non-zero-offset.t
SECTIONS {
  .text : { *(.text .text.*) }
  .eh_frame : {
    /* Padding within .eh_frame */
    . += 128;
    __eh_frame_start = .;
    *(.eh_frame) ;
    __eh_frame_end = .;
  }
}

//--- eh-frame-zero-offset.t
SECTIONS {
  .text : { *(.text .text.*) }
  .eh_frame : {
    __eh_frame_start = .;
    *(.eh_frame) ;
    __eh_frame_end = .;
  }
}

//--- a.s
.section .text.01, "ax",%progbits
.global f1
.type f1, %function
f1:
.cfi_startproc
.space 4
.cfi_endproc
