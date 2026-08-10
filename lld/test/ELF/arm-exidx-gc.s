// REQUIRES: arm
// RUN: llvm-mc -filetype=obj --arm-add-build-attributes -triple=armv7a-none-linux-gnueabi %s -o %t
// RUN: ld.lld %t --no-merge-exidx-entries -o %t2 --gc-sections
// RUN: llvm-objdump -d --triple=armv7a-none-linux-gnueabi --no-show-raw-insn %t2 | FileCheck %s
// RUN: llvm-objdump -s --triple=armv7a-none-linux-gnueabi %t2 | FileCheck --check-prefix=CHECK-EXIDX %s

/// Test the behavior of .ARM.exidx sections under garbage collection
/// A .ARM.exidx section is live if it has a relocation to a live executable
/// section.
/// A .ARM.exidx section may have a relocation to a .ARM.extab section, if the
/// .ARM.exidx is live then the .ARM.extab section is live

 .syntax unified
 .section .text.func1, "ax",%progbits
 .global func1
func1:
 .fnstart
 bx lr
 .save {r7, lr}
 .setfp r7, sp, #0
 .fnend

 .section .text.unusedfunc1, "ax",%progbits
 .global unusedfunc1
unusedfunc1:
 .fnstart
 bx lr
 .cantunwind
 .fnend

 /// Unwinding instructions for .text2 too large for an inline entry ARM.exidx
 /// entry. A separate .ARM.extab section is created to hold the unwind entries
 /// The .ARM.exidx table entry has a reference to the .ARM.extab section.
 .section .text.func2, "ax",%progbits
 .global func2
func2:
 .fnstart
 bx lr
 .personality __gxx_personality_v0
 .handlerdata
 .section .text.func2
 .fnend

 /// An unused function with a reference to a .ARM.extab section. Both should
 /// be removed by gc.
 .section .text.unusedfunc2, "ax",%progbits
 .global unusedfunc2
unusedfunc2:
 .fnstart
 bx lr
 .personality __gxx_personality_v1
 .handlerdata
 .section .text.unusedfunc2
 .fnend

 /// Dummy implementation of personality routines to satisfy reference from
 /// exception tables
 .section .text.__gcc_personality_v0, "ax", %progbits
 .global __gxx_personality_v0
__gxx_personality_v0:
 .fnstart
 bx lr
 .cantunwind
 .fnend

 .section .text.__gcc_personality_v1, "ax", %progbits
 .global __gxx_personality_v1
__gxx_personality_v1:
 .fnstart
 bx lr
 .cantunwind
 .fnend

 .section .text.__aeabi_unwind_cpp_pr0, "ax", %progbits
 .global __aeabi_unwind_cpp_pr0
__aeabi_unwind_cpp_pr0:
 .fnstart
 bx lr
 .cantunwind
 .fnend

// Entry point for GC
 .text
 .global _start
_start:
 bl func1
 bl func2
 bx lr

/// GC should have only removed unusedfunc1 and unusedfunc2 the personality
/// routines are kept alive by references from live .ARM.exidx and .ARM.extab
/// sections
// CHECK: Disassembly of section .text:
// CHECK-EMPTY:
// CHECK-NEXT: <_start>:
// CHECK-NEXT:   2010c:       bl      0x20118 <func1>
// CHECK-NEXT:   20110:       bl      0x2011c <func2>
// CHECK-NEXT:   20114:       bx      lr
// CHECK: <func1>:
// CHECK-NEXT:   20118:       bx      lr
// CHECK: <func2>:
// CHECK-NEXT:   2011c:       bx      lr
// CHECK: <__gxx_personality_v0>:
// CHECK-NEXT:   20120:       bx      lr
// CHECK: <__aeabi_unwind_cpp_pr0>:
// CHECK-NEXT:   20124:       bx      lr

/// GC should have removed table entries for unusedfunc1, unusedfunc2
/// and __gxx_personality_v1
// CHECK-NOT: unusedfunc1
// CHECK-NOT: unusedfunc2
// CHECK-NOT: __gxx_personality_v1

/// CHECK-EXIDX: Contents of section .ARM.exidx:
/// 100d4 + 1038 = 1110c = _start
/// 100dc + 103c = 11118 = func1
// CHECK-EXIDX-NEXT: 100d4 38000100 01000000 3c000100 08849780
/// 100e4 + 1038 = 1111c = func2 (100e8 + 1c = 10104 = .ARM.extab)
/// 100ec + 1034 = 11120 = __gxx_personality_v0
// CHECK-EXIDX-NEXT: 100e4 38000100 1c000000 34000100 01000000
/// 100f4 + 1030 = 11018 = __aeabi_unwind_cpp_pr0
/// 100fc + 102c = 1101c = __aeabi_unwind_cpp_pr0 + sizeof(__aeabi_unwind_cpp_pr0)
// CHECK-EXIDX-NEXT: 100f4 30000100 01000000 2c000100 01000000
// CHECK-EXIDX-NEXT: Contents of section .ARM.extab:
/// 10104 + 101c = 11120 = __gxx_personality_v0
// CHECK-EXIDX-NEXT: 10104 1c000100 b0b0b000
