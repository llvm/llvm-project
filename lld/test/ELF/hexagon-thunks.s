# REQUIRES: hexagon
# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf %t/a.s -o %t/a.o
# RUN: ld.lld -T %t/lds %t/a.o -o %t/a
# RUN: llvm-objdump -d --no-show-raw-insn %t/a 2>&1 | \
# RUN:     FileCheck --check-prefixes=CHECK-NONPIC,CHECK %s
# RUN: llvm-mc -filetype=obj \
# RUN:         -triple=hexagon-unknown-elf %t/a.s -o %t/a.o

# RUN: ld.lld -T %t/lds --pie %t/a.o -o %t/a
# RUN: llvm-objdump -d --no-show-raw-insn %t/a 2>&1 | \
# RUN:     FileCheck --check-prefixes=CHECK-PIC,CHECK %s

#--- a.s
.section .text_low, "ax", %progbits
    .globl main
    .type  main, @function
main:
    call myfn
    jumpr r31
    .size   main, .-main

.section .text_high, "ax", %progbits
    .globl myfn
    .type  myfn, @function
myfn:
    jumpr r31
    .size  myfn, .-myfn

# CHECK:  Disassembly of section .text_low:

# CHECK:             <__hexagon_thunk_myfn_from_.text.thunk>:
# CHECK-NONPIC-NEXT:    200b4: { immext(#0x1000000)
# CHECK-NONPIC-NEXT:             jump 0x10200bc <myfn> }
# CHECK-PIC-NEXT:       200b4: { immext(#0x1000000)
# CHECK-PIC-NEXT:               r14 = add(pc,##0x1000008) }
# CHECK-PIC-NEXT:       { jumpr r14 }

# CHECK-NONPIC:      <main>:
# CHECK-NONPIC-NEXT:   200bc: { call 0x200b4 <__hexagon_thunk_myfn_from_.text.thunk> }
# CHECK-PIC:         <main>:
# CHECK-PIC-NEXT:      200c0: { call 0x200b4 <__hexagon_thunk_myfn_from_.text.thunk> }
# CHECK-NEXT:                 { jumpr r31 }

# CHECK:  Disassembly of section .text_high:
# CHECK:    <myfn>:
# CHECK-NEXT: 10200bc: { jumpr r31 }

#--- lds
SECTIONS {
  .text_low 0x200b4: { *(.text_low) }
  .text_high 0x10200bc : { *(.text_high) }
}
