# REQUIRES: hexagon
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf main.s -o main.o
# RUN: ld.lld -shared -T lds main.o -o test.so
# RUN: llvm-objdump -d --no-show-raw-insn test.so | FileCheck %s

## Test thunk range scenarios for Hexagon R_HEX_GD_PLT_B22_PCREL relocations.
## Same ±8MB range as regular calls.

#--- main.s
.globl _start
.type _start, @function
_start:
  ## Setup for TLS Global Dynamic calls
  r2 = add(pc,##_GLOBAL_OFFSET_TABLE_@PCREL)

  ## Test TLS GD PLT calls
  r0 = add(r2,##tls_var_close@GDGOT)
  call tls_var_close@GDPLT

  r0 = add(r2,##tls_var_far@GDGOT)
  call tls_var_far@GDPLT

  jumpr r31

.section .text_far,"ax",@progbits
more_code:
  r0 = add(r2,##tls_var_distant@GDGOT)
  call tls_var_distant@GDPLT
  jumpr r31

## TLS variables in .tdata section
.section .tdata,"awT",@progbits
.globl tls_var_close, tls_var_far, tls_var_distant
.type tls_var_close, @object
.type tls_var_far, @object
.type tls_var_distant, @object

tls_var_close:
  .word 0x1234

tls_var_far:
  .word 0x5678

tls_var_distant:
  .word 0x9abc

#--- lds
SECTIONS {
  .text 0x10000 : { *(.text) }
  .text_far 0x410000 : { *(.text_far) }
}

# CHECK: Disassembly of section .text:
# CHECK:     <_start>:
# CHECK-NEXT:   10000:  { immext(#0x4000c0)
# CHECK-NEXT:      r2 = add(pc,##0x4000d4) }
# CHECK-NEXT:    { immext(#0xffffffc0)
# CHECK-NEXT:      r0 = add(r2,##-0x18) }
# CHECK-NEXT:    { call 0x410030 <__tls_get_addr@plt> }
# CHECK-NEXT:    { immext(#0xffffffc0)
# CHECK-NEXT:      r0 = add(r2,##-0x10) }
# CHECK-NEXT:    { call 0x410030 <__tls_get_addr@plt> }
# CHECK-NEXT:    { jumpr r31 }

# CHECK:     <more_code>:
# CHECK-NEXT:   410000:  { immext(#0xffffffc0)
# CHECK-NEXT:      r0 = add(r2,##-0x8) }
# CHECK-NEXT:    { call 0x410030 <__tls_get_addr@plt> }
# CHECK-NEXT:    { jumpr r31 }

## Verify PLT entry is created for __tls_get_addr
## TLS symbols (tls_var_close, tls_var_far, tls_var_distant) should NOT have
## PLT entries - only __tls_get_addr needs one.
# CHECK: Disassembly of section .plt:
# CHECK:      <.plt>:
# CHECK-NEXT:   410010:  { immext(#0xc0)
# CHECK-NEXT:      r28 = add(pc,##0xc4) }
# CHECK-NEXT:    { r14 -= add(r28,#0x10)
# CHECK-NEXT:      r15 = memw(r28+#0x8)
# CHECK-NEXT:      r28 = memw(r28+#0x4) }
# CHECK-NEXT:    { r14 = asr(r14,#0x2)
# CHECK-NEXT:      jumpr r28 }
# CHECK-NEXT:    { trap0(#0xdb) }

# CHECK:      <__tls_get_addr@plt>:
# CHECK-NEXT:   410030: { immext(#0x80)
# CHECK-NEXT:      r14 = add(pc,##0xb4) }
# CHECK-NEXT:    { r28 = memw(r14+#0x0) }
# CHECK-NEXT:    { jumpr r28 }
