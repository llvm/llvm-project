// REQUIRES: x86
// RUN: rm -rf %t && split-file %s %t

// RUN: llvm-mc -filetype=obj -triple=x86_64 %t/a.s -o %t/a.o
// RUN: ld.lld %t/a.o -T %t/eh-frame-non-zero-offset.t -o %t/non-zero
// RUN: llvm-readelf --program-headers --unwind --symbols -x .eh_frame %t/non-zero | FileCheck --check-prefix=NONZERO %s
// RUN: ld.lld %t/a.o -T %t/eh-frame-zero-offset.t -o %t/zero
// RUN: llvm-readelf --program-headers --unwind --symbols -x .eh_frame %t/zero | FileCheck --check-prefix=ZERO %s

// NONZERO:      {{[0-9]+}}: 0000000000000080 {{.*}} __eh_frame_start
// NONZERO-NEXT: {{[0-9]+}}: 00000000000000b4 {{.*}} __eh_frame_end

// NONZERO:      0x00000078 00000000 00000000 14000000 00000000
// NONZERO-NEXT: 0x00000088 017a5200 01781001 1b0c0708 90010000
// NONZERO-NEXT: 0x00000098 14000000 1c000000 60ffffff 04000000
// NONZERO-NEXT: 0x000000a8 00000000 00000000 00000000

// ZERO:      {{[0-9]+}}: 0000000000000080 {{.*}} __eh_frame_start
// ZERO-NEXT: {{[0-9]+}}: 00000000000000b4 {{.*}} __eh_frame_end

// ZERO:      0x00000080 14000000 00000000 017a5200 01781001
// ZERO-NEXT: 0x00000090 1b0c0708 90010000 14000000 1c000000
// ZERO-NEXT: 0x000000a0 60ffffff 04000000 00000000 00000000
// ZERO-NEXT: 0x000000b0 00000000

//--- eh-frame-non-zero-offset.t
SECTIONS {
  .text : { *(.text .text.*) }
  .eh_frame : {
  /* Alignment padding within .eh_frame */
  . = ALIGN(128);
  __eh_frame_start = .;
  *(.eh_frame .eh_frame.*) ;
  __eh_frame_end = .;
  }
}

//--- eh-frame-zero-offset.t
SECTIONS {
  .text : { *(.text .text.*) }
  .eh_frame : ALIGN(128) {
  __eh_frame_start = .;
  *(.eh_frame .eh_frame.*) ;
  __eh_frame_end = .;
  }
}

//--- a.s
.section .text
.globl f1
f1:
.cfi_startproc
.space 4
.cfi_endproc
