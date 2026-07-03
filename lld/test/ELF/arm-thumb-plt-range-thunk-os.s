// REQUIRES: arm
// RUN: rm -rf %t && split-file %s %t && cd %t
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=thumbv7a-none-linux-gnueabi a.s -o a.o
// RUN: ld.lld a.o --shared --icf=all -o a.so --script=a.lds --print-map --print-icf-sections
// RUN: llvm-objdump --no-show-raw-insn --no-print-imm-hex -d a.so | FileCheck %s
// RUN: rm a.so

//--- a.lds
SECTIONS {
  .dynsym 0x1ff0000 : AT(0x1ff0000) { *(.dynsym) }
  .text.1 0x2000000 : AT(0x2000000) { *(.text) *(.text.1) }
  .text.2 0x4000000 : AT(0x4000000) { *(.text.2) *(.text.expect.icf) }
  .plt : { *(.plt) }
}

//--- a.s

 .syntax unified
 .thumb

// Make sure that we generate a range extension thunk to a PLT entry
 .section ".text.1", "ax", %progbits
 .global sym1
 .global elsewhere
 .type elsewhere, %function
 .global preemptible
 .type preemptible, %function
 .global far_preemptible
 .type far_preemptible, %function
 .global far_nonpreemptible
 .hidden far_nonpreemptible
 .type far_nonpreemptible, %function
 .global far_nonpreemptible_alias
 .hidden far_nonpreemptible_alias
 .type far_nonpreemptible_alias, %function
sym1:
 bl elsewhere
 bl preemptible
 bx lr
preemptible:
 bl far_preemptible
 bl far_nonpreemptible
 bl far_nonpreemptible_alias
bx lr

// CHECK-LABEL: <sym1>:
// CHECK-NEXT: 2000000: bl 0x2000018 <__ThumbV7PILongThunk_elsewhere>
// CHECK-NEXT:          bl 0x2000024 <__ThumbV7PILongThunk_preemptible>
// CHECK-NEXT:          bx lr

// CHECK-LABEL: <preemptible>:
// CHECK-NEXT: 200000a: bl 0x2000030 <__ThumbV7PILongThunk_far_preemptible>
// CHECK-NEXT:          bl 0x200003c <__ThumbV7PILongThunk_far_nonpreemptible>
// CHECK-NEXT:          bl 0x200003c <__ThumbV7PILongThunk_far_nonpreemptible>
// CHECK-NEXT:          bx lr

// CHECK-LABEL: <__ThumbV7PILongThunk_elsewhere>:
// CHECK-NEXT: 2000018: movw    r12, #12
// CHECK-NEXT:          movt    r12, #512
// CHECK-NEXT:          add     r12, pc
// CHECK-NEXT:          bx      r12

// CHECK-LABEL: <__ThumbV7PILongThunk_preemptible>:
// CHECK-NEXT: 2000024: movw    r12, #16
// CHECK-NEXT:          movt    r12, #512
// CHECK-NEXT:          add     r12, pc
// CHECK-NEXT:          bx      r12

// CHECK-LABEL: <__ThumbV7PILongThunk_far_preemptible>:
// CHECK-NEXT: 2000030: movw    r12, #20
// CHECK-NEXT:          movt    r12, #512
// CHECK-NEXT:          add     r12, pc
// CHECK-NEXT:          bx      r12

// CHECK-LABEL: <__ThumbV7PILongThunk_far_nonpreemptible>:
// CHECK-NEXT: 200003c: movw    r12, #65465
// CHECK-NEXT:          movt    r12, #511
// CHECK-NEXT:          add     r12, pc
// CHECK-NEXT:          bx      r12

 .section .text.2, "ax", %progbits
far_preemptible:
far_nonpreemptible:
 bl elsewhere

 .section .text.expect.icf, "ax", %progbits
far_nonpreemptible_alias:
 bl elsewhere

// CHECK-LABEL: <far_preemptible>:
// CHECK-NEXT:  4000000:         blx     0x4000030 <elsewhere@plt>

// CHECK-LABEL: <elsewhere@plt>:
// CHECK-NEXT:  4000030:          add     r12, pc, #0, #12

// CHECK-LABEL: <preemptible@plt>:
// CHECK-NEXT:  4000040:          add     r12, pc, #0, #12

// CHECK-LABEL: <far_preemptible@plt>:
// CHECK-NEXT:  4000050:          add     r12, pc, #0, #12
