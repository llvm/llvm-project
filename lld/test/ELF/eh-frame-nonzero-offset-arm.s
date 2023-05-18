// REQUIRES: arm
// RUN: rm -rf %t && split-file %s %t && cd %t

// RUN: llvm-mc -filetype=obj -triple=arm a.s -o a.o
// RUN: ld.lld a.o -T eh-frame-non-zero-offset.t -o non-zero
// RUN: llvm-readelf --program-headers --unwind --symbols -x .eh_frame non-zero | FileCheck --check-prefix=NONZERO %s
// RUN: ld.lld a.o -T eh-frame-zero-offset.t -o zero
// RUN: llvm-readelf --program-headers --unwind --symbols -x .eh_frame zero | FileCheck --check-prefix=ZERO %s

// NONZERO:      {{[0-9]+}}: 00000084 {{.*}} __eh_frame_start
// NONZERO-NEXT: {{[0-9]+}}: 000000b0 {{.*}} __eh_frame_end

// NONZERO:      0x00000084 10000000 00000000 017a5200 017c0e01
// NONZERO-NEXT: 0x00000094 1b0c0d00 10000000 18000000 60ffffff
// NONZERO-NEXT: 0x000000a4 04000000 00000000 00000000

// ZERO:      {{[0-9]+}}: 00000004 {{.*}} __eh_frame_start
// ZERO-NEXT: {{[0-9]+}}: 00000030 {{.*}} __eh_frame_end

// ZERO:      0x00000004 10000000 00000000 017a5200 017c0e01
// ZERO-NEXT: 0x00000014 1b0c0d00 10000000 18000000 e0ffffff
// ZERO-NEXT: 0x00000024 04000000 00000000 00000000

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
.cfi_sections .eh_frame
.space 4
.cfi_endproc
