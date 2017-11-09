# Check less-significant bit setup for microMIPS PLT.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -mattr=micromips %S/Inputs/mips-dynamic.s -o %t-dso.o
# RUN: ld.lld %t-dso.o -shared -o %t.so
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -mattr=micromips %s -o %t-exe.o
# RUN: ld.lld %t-exe.o %t.so -o %t.exe
# RUN: llvm-readobj -mips-plt-got %t.exe | FileCheck %s

# REQUIRES: mips

# CHECK:      Primary GOT {
# CHECK:        Local entries [
# CHECK-NEXT:     Entry {
# CHECK-NEXT:       Address:
# CHECK-NEXT:       Access:
# CHECK-NEXT:       Initial: 0x20009
# CHECK-NEXT:     }
# CHECK:        ]
# CHECK:      }

# CHECK:      PLT GOT {
# CHECK:        Entries [
# CHECK-NEXT:     Entry {
# CHECK-NEXT:       Address:
# CHECK-NEXT:       Initial: 0x20011
# CHECK-NEXT:       Value: 0x0
# CHECK-NEXT:       Type: Function
# CHECK-NEXT:       Section: Undefined
# CHECK-NEXT:       Name: foo0@
# CHECK-NEXT:     }
# CHECK-NEXT:   ]
# CHECK-NEXT: }

  .text
  .set micromips
  .global foo
  .hidden foo
  .global __start
__start:
  lw    $t0,%got(foo)($gp)
  addi  $t0,$t0,%lo(foo)
foo:
  jal   foo0
