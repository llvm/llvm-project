# REQUIRES: x86
## PR48357: If .rela.dyn appears as an output section description, its type may
## be SHT_RELA (due to the empty synthetic .rela.plt) while there is no input
## section. The empty .rela.dyn may be retained due to a reference. Don't crash.

# RUN: llvm-mc -filetype=obj -triple=x86_64 /dev/null -o %t.o
# RUN: ld.lld -shared --emit-relocs -T %s %t.o -o %t
# RUN: llvm-readelf -S %t | FileCheck %s

## Note, sh_link of such an empty .rela.dyn is 0.
# CHECK: Name      Type Address          Off    Size   ES Flg Lk Inf Al
# CHECK: .rela.dyn RELA 0000000000000000 001000 000000 18   A  0   0  8

SECTIONS {
  .rela.dyn : { *(.rela*) }
  __rela_offset = ABSOLUTE(ADDR(.rela.dyn));
}
