// REQUIRES: x86
// RUN: rm -rf %t && split-file %s %t && cd %t

// RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
// RUN: ld.lld a.o -T eh-frame-non-zero-offset.t -o non-zero
// RUN: llvm-readelf --program-headers --unwind --symbols -x .eh_frame non-zero | FileCheck --check-prefix=NONZERO %s
// RUN: ld.lld a.o -T eh-frame-zero-offset.t -o zero
// RUN: llvm-readelf --program-headers --unwind --symbols -x .eh_frame zero | FileCheck --check-prefix=ZERO %s

// NONZERO:      {{[0-9]+}}: 0000000000000088 {{.*}} __eh_frame_start
// NONZERO-NEXT: {{[0-9]+}}: 00000000000000bc {{.*}} __eh_frame_end

// NONZERO:      0x00000088 14000000 00000000 017a5200 01781001
// NONZERO-NEXT: 0x00000098 1b0c0708 90010000 14000000 1c000000
// NONZERO-NEXT: 0x000000a8 58ffffff 04000000 00000000 00000000
// NONZERO-NEXT: 0x000000b8 00000000

// ZERO:      {{[0-9]+}}: 0000000000000008 {{.*}} __eh_frame_start
// ZERO-NEXT: {{[0-9]+}}: 000000000000003c {{.*}} __eh_frame_end

// ZERO:      0x00000008 14000000 00000000 017a5200 01781001
// ZERO-NEXT: 0x00000018 1b0c0708 90010000 14000000 1c000000
// ZERO-NEXT: 0x00000028 d8ffffff 04000000 00000000 00000000
// ZERO-NEXT: 0x00000038 00000000

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
