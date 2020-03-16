// REQUIRES: arm
// RUN: llvm-mc -filetype=obj --arm-add-build-attributes -triple=armv7a-none-linux-gnueabi %s -o %t
// RUN: ld.lld %t -o %t2
// RUN: llvm-objdump -d --triple=armv7a-none-linux-gnueabi --no-show-raw-insn %t2 | FileCheck %s
// RUN: llvm-objdump -s --triple=armv7a-none-linux-gnueabi %t2 | FileCheck --check-prefix=CHECK-EXIDX %s
// RUN: llvm-readobj --program-headers --sections %t2 | FileCheck -check-prefix=CHECK-PT %s

// Test that inline unwinding table entries and references to .ARM.extab
// entries survive the re-ordering of the .ARM.exidx section

 .syntax unified
 // Will produce an ARM.exidx entry with inline unwinding instructions
 .section .text.func1, "ax",%progbits
 .global func1
func1:
 .fnstart
 bx lr
 .save {r7, lr}
 .setfp r7, sp, #0
 .fnend

 // Unwinding instructions for .text2 too large for an inline entry ARM.exidx
 // entry. A separate .ARM.extab section is created to hold the unwind entries
 // The .ARM.exidx table entry has a reference to the .ARM.extab section.
 .section .text.func2, "ax",%progbits
 .global func2
func2:
 .fnstart
 bx lr
 .personality __gxx_personality_v0
 .handlerdata
 .long 0
 .section .text.func2
 .fnend

 // Dummy implementation of personality routines to satisfy reference from
 // exception tables
 .section .text.__gcc_personality_v0, "ax", %progbits
 .global __gxx_personality_v0
__gxx_personality_v0:
 bx lr

 .section .text.__aeabi_unwind_cpp_pr0, "ax", %progbits
 .global __aeabi_unwind_cpp_pr0
__aeabi_unwind_cpp_pr0:
 bx lr

 .text
 .global _start
_start:
 bl func1
 bl func2
 bx lr

// CHECK: Disassembly of section .text:
// CHECK-EMPTY:
// CHECK-NEXT: <_start>:
// CHECK-NEXT:    11108:       bl      #4 <func1>
// CHECK-NEXT:                 bl      #4 <func2>
// CHECK-NEXT:                 bx      lr
// CHECK:      <func1>:
// CHECK-NEXT:    11114:       bx      lr
// CHECK:      <func2>:
// CHECK-NEXT:    11118:       bx      lr
// CHECK:      <__gxx_personality_v0>:
// CHECK-NEXT:    1111c:       bx      lr
// CHECK:      <__aeabi_unwind_cpp_pr0>:
// CHECK-NEXT:    11120:       bx      lr

// 100d4 + 0x1034 = 0x11108 = main (linker generated cantunwind)
// 100dc + 0x1038 = 0x11114 = func1 (inline unwinding data)
// CHECK-EXIDX:      100d4 34100000 01000000 38100000 08849780
// 100e4 + 0x1034 = 0x11118 = func2 (100e8 + 14 = 100fc = .ARM.extab entry)
// 100ec + 0x1030 = 0x1111c = __gxx_personality_v0 (linker generated cantunwind)
// CHECK-EXIDX-NEXT: 100e4 34100000 14000000 30100000 01000000
// 100f4 + 0x1030 = 1101c = sentinel
// CHECK-EXIDX-NEXT: 100f4 30100000 01000000

// CHECK-PT:          Name: .ARM.exidx
// CHECK-PT-NEXT:     Type: SHT_ARM_EXIDX (0x70000001)
// CHECK-PT-NEXT:     Flags [
// CHECK-PT-NEXT:       SHF_ALLOC
// CHECK-PT-NEXT:       SHF_LINK_ORDER
// CHECK-PT-NEXT:     ]
// CHECK-PT-NEXT:     Address: 0x100D4
// CHECK-PT-NEXT:     Offset: 0xD4
// CHECK-PT-NEXT:     Size: 40

// CHECK-PT:          Type: PT_ARM_EXIDX (0x70000001)
// CHECK-PT-NEXT:     Offset: 0xD4
// CHECK-PT-NEXT:     VirtualAddress: 0x100D4
// CHECK-PT-NEXT:     PhysicalAddress: 0x100D4
// CHECK-PT-NEXT:     FileSize: 40
// CHECK-PT-NEXT:     MemSize: 40
// CHECK-PT-NEXT:     Flags [ (0x4)
// CHECK-PT-NEXT:       PF_R (0x4)
// CHECK-PT-NEXT:     ]
// CHECK-PT-NEXT:     Alignment: 4
// CHECK-PT-NEXT:   }
