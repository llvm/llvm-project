## Recover an unnamed Cortex-A53 erratum 843419 veneer. The ADRP sits at
## page offset 0xffc. An intervening load follows the address dependency while
## redefining the register, and the veneer uses the resulting value before
## returning to the instruction after the redirected branch.
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-linux %t/input.s -o %t.o
# RUN: ld.lld -T %t/layout.ld %t.o -o %t.exe
# RUN: llvm-strip --strip-all %t.exe -o %t.stripped
# RUN: llvm-bolt %t.stripped -o %t.bolt --allow-stripped \
# RUN:   --recover-relocations --print-disasm \
# RUN:   --print-only=__BOLT_e843419_212000 2>&1 | FileCheck %s

# CHECK: Binary Function "__BOLT_e843419_212000"
# CHECK: ldr x1, [x0]
# CHECK-NEXT: b

#--- input.s
.section .text.source,"ax",%progbits
.globl source
.hidden source
.type source, %function
source:
.cfi_startproc
  .space 0xffc
  adrp x0, pointer
  ldr x0, [x0, :lo12:pointer]
  b veneer
.Lreturn:
  ret
.cfi_endproc
.size source, .-source

.section .text.veneer,"ax",%progbits
.type veneer, %function
veneer:
  ldr x1, [x0, :lo12:object]
  b .Lreturn
.size veneer, .-veneer

.data
.p2align 3
object:
  .xword 42
pointer:
  .xword object

#--- layout.ld
ENTRY(source)
SECTIONS {
  . = 0x210000;
  .text : { *(.text.source) }
  . = 0x212000;
  .text.veneer : { *(.text.veneer) }
  . = 0x220000;
  .data : { *(.data) }
  .eh_frame : { *(.eh_frame) }
}
