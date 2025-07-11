# REQUIRES: hexagon
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf a.s -o a.o
# RUN: ld.lld -shared a.o -soname a.so -o a.so
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf b.s -o b.o
# RUN: ld.lld b.o a.so -o out
# RUN: llvm-objdump -d --no-show-raw-insn out | FileCheck %s

## Test thunk range scenarios for Hexagon R_HEX_PLT_B22_PCREL relocations.
## PLT calls use the same ±8MB range as regular calls but go through PLT entries.
## This test verifies thunk generation for PLT calls at range boundaries.

#--- a.s
.globl extern_within_range, extern_beyond_range, extern_close
.type extern_within_range, @function
.type extern_beyond_range, @function
.type extern_close, @function

extern_within_range:
  jumpr r31

extern_beyond_range:
  jumpr r31

extern_close:
  jumpr r31

#--- b.s
.globl _start
.type _start, @function
_start:
  ## Test PLT calls to external functions at various ranges
  call extern_within_range@PLT
  call extern_beyond_range@PLT
  call extern_close@PLT
  jumpr r31

.skip 0x200000

# CHECK: Disassembly of section .text:
# CHECK:     <_start>:
# CHECK-NEXT:  20214:   { call 0x220250 }
# CHECK-NEXT:    { call 0x220260 }
# CHECK-NEXT:    { call 0x220270 }
# CHECK-NEXT:    { jumpr r31 }

## Verify PLT header and entries are created with exact addresses
# CHECK: Disassembly of section .plt:
# CHECK:      <.plt>:
# CHECK-NEXT:   220230:    { immext(#0x20080)
# CHECK-NEXT:      r28 = add(pc,##0x200b8) }
# CHECK-NEXT:    { r14 -= add(r28,#0x10)
# CHECK-NEXT:      r15 = memw(r28+#0x8)
# CHECK-NEXT:      r28 = memw(r28+#0x4) }
# CHECK-NEXT:    { r14 = asr(r14,#0x2)
# CHECK-NEXT:      jumpr r28 }
# CHECK-NEXT:    { trap0(#0xdb) }

# CHECK:     <extern_within_range@plt>:
# CHECK-NEXT:  220250:   { immext(#0x20080)
# CHECK-NEXT:      r14 = add(pc,##0x200a8) }
# CHECK-NEXT:    { r28 = memw(r14+#0x0) }
# CHECK-NEXT:    { jumpr r28 }

# CHECK:     <extern_beyond_range@plt>:
# CHECK-NEXT:  220260:   { immext(#0x20080)
# CHECK-NEXT:      r14 = add(pc,##0x2009c) }
# CHECK-NEXT:    { r28 = memw(r14+#0x0) }
# CHECK-NEXT:    { jumpr r28 }

# CHECK:     <extern_close@plt>:
# CHECK-NEXT:  220270:   { immext(#0x20080)
# CHECK-NEXT:      r14 = add(pc,##0x20090) }
# CHECK-NEXT:    { r28 = memw(r14+#0x0) }
# CHECK-NEXT:    { jumpr r28 }
