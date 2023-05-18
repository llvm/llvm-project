// REQUIRES: arm
// RUN: rm -rf %t && split-file %s %t

// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi --arm-add-build-attributes %t/a.s -o %t/a.o
// RUN: ld.lld %t/a.o -T %t/exidx-non-zero-offset.t -o %t/non-zero
// RUN: llvm-readelf --program-headers --unwind --symbols -x .exceptions %t/non-zero | FileCheck %s
// RUN: ld.lld %t/a.o -T %t/exidx-zero-offset.t -o %t/zero
// RUN: llvm-readelf --program-headers --unwind --symbols -x .exceptions %t/zero | FileCheck %s

/// On platforms that load ELF files directly the ARM.exidx sections
/// are located with the PT_ARM_EXIDX program header. This requires
/// all .ARM.exidx input sections to be placed in a single
/// .ARM.exidx output section. Embedded systems that do not load
/// from an ELF file use the linker defined symbols __exidx_start and
/// __exidx_stop. There is no requirement to place the .ARM.exidx
/// input sections in their own output section. This test case checks
/// that a .ARM.exidx synthetic section that isn't at a zero offset
/// within the output section gets the correct offsets. We check
/// this by checking that equal amounts of alignment padding
/// inserted before and after the section start produce the same
/// results.

/// For the two linker scripts, one with the .ARM.exidx table
/// at a zero offset within its output section and one with a
/// non-zero offset, but both at the same file and address;
/// the PT_ARM_EXIDX, symbols and table contents should be the
/// same.
// CHECK:      EXIDX 0x010080 0x00000080 0x00000080 0x00028 0x00028 R   0x4

/// unwind entries starting from the right address are identical.
/// llvm-readelf does not seem to be able to detect Thumb functionNames
/// whereas arm-none-eabi-readelf can. Used CHECK rather than CHECK-NEXT
/// to cope with possible improvements llvm-readelf.
// CHECK:        FunctionAddress: 0x0
// CHECK-NEXT:   FunctionName: f1
// CHECK-NEXT:   Model: Compact (Inline)
// CHECK-NEXT:   PersonalityIndex: 0
// CHECK-NEXT:   Opcodes [
// CHECK-NEXT:     0x97      ; vsp = r7
// CHECK-NEXT:     0x84 0x08 ; pop {r7, lr}
// CHECK-NEXT:   ]
// CHECK-NEXT: }
// CHECK-NEXT: Entry {
// CHECK-NEXT:   FunctionAddress: 0x8
// CHECK-NEXT:   FunctionName: f2
// CHECK-NEXT:   Model: CantUnwind
// CHECK-NEXT: }
// CHECK-NEXT: Entry {
// CHECK-NEXT:   FunctionAddress: 0xE
// CHECK:        ExceptionHandlingTable: .ARM.extab
// CHECK-NEXT:   TableEntryAddress: 0xA8
// CHECK-NEXT:   Model: Generic
// CHECK-NEXT:   PersonalityRoutineAddress: 0x12
// CHECK-NEXT: }
// CHECK-NEXT: Entry {
// CHECK-NEXT:   FunctionAddress: 0x10
// CHECK:        Model: CantUnwind
// CHECK-NEXT: }
// CHECK-NEXT: Entry {
// CHECK-NEXT:   FunctionAddress: 0x14
// CHECK-NEXT:   FunctionName: __ARMv7ABSLongThunk_f3
// CHECK-NEXT:   Model: CantUnwind
// CHECK-NEXT: }

// CHECK:      {{[0-9]+}}: 00000080 {{.*}} __exidx_start
// CHECK-NEXT: {{[0-9]+}}: 000000a8 {{.*}} __exidx_end

// CHECK:      0x00000080 80ffff7f 08849780 80ffff7f 01000000
// CHECK-NEXT: 0x00000090 7effff7f 14000000 78ffff7f 01000000
// CHECK-NEXT: 0x000000a0 74ffff7f 01000000

//--- exidx-non-zero-offset.t
SECTIONS {
  .text : { *(.text .text.*) }
  /* Addition of thunk in .text changes alignment padding */
  .exceptions : {
  /* Alignment padding within .exceptions */
  . = ALIGN(128);
  /* Embedded C libraries find exceptions via linker defined
     symbols */
  __exidx_start = .;
  *(.ARM.exidx) ;
  __exidx_end = .;
  }
  .ARM.extab : { *(.ARM.extab .ARM.extab.*) }
}

//--- exidx-zero-offset.t
SECTIONS {
  .text : { *(.text .text.*) }
  /* Addition of thunk in .text changes alignment padding */
  /* alignment padding before .exceptions starts */
  .exceptions : ALIGN(128) {
  /* Embedded C libraries find exceptions via linker defined
     symbols */
  __exidx_start = .;
  *(.ARM.exidx) ;
  __exidx_end = .;
  }
  .ARM.extab : { *(.ARM.extab .ARM.extab.*) }
}

//--- a.s
.syntax unified

/// Expect inline unwind instructions.
.section .text.01, "ax", %progbits
.arm
.balign 4
.global f1
.type f1, %function
f1:
.fnstart
/// provoke an interworking thunk.
b f3
bx lr
.save {r7, lr}
.setfp r7, sp, #0
.fnend

/// Expect no unwind information from assembler. The linker must
/// synthesise an EXIDX_CANTUNWIND entry to prevent an exception
/// thrown through f2 from matching against the unwind instructions
/// for f1.
.section .text.02, "ax", %progbits
.global f2
.type f2, %function
.balign 4
f2:
bx lr

/// Expect 1 EXIDX_CANTUNWIND entry that can be merged into the linker
/// generated EXIDX_CANTUNWIND as if the assembler had generated it.
.section .text.03, "ax",%progbits
.global f3
.type f3, %function
.thumb
.balign 2
f3:
.fnstart
bx lr
.cantunwind
.fnend

/// Expect a section with a reference to an .ARM.extab.
.section .text.04, "ax",%progbits
.global f4
.balign 2
.type f4, %function
f4:
.fnstart
bx lr
.personality __gxx_personality_v0
.handlerdata
.long 0
.fnend


/// Dummy implementation of personality routines to satisfy reference
/// from exception tables, linker will generate EXIDX_CANTUNWIND.
.section .text.__aeabi_unwind_cpp_pr0, "ax", %progbits
.global __aeabi_unwind_cpp_pr0
.type __aeabi_unwind_cpp_pr0, %function
.balign 2
__aeabi_unwind_cpp_pr0:
bx lr

.section .text.__gcc_personality_v0, "ax", %progbits
.global __gxx_personality_v0
.type __gcc_personality_v0, %function
.balign 2
__gxx_personality_v0:
bx lr
