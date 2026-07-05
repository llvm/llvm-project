# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/a.s -o %t/a.o

## Discard .plt, .iplt, and .got.plt
# RUN: ld.lld -shared -T %t/t %t/a.o -o %t/a
# RUN: llvm-readelf -S -d %t/a > %t/readelf.txt
# RUN: FileCheck %s --input-file %t/readelf.txt
# RUN: FileCheck %s --input-file %t/readelf.txt --check-prefix=NEG

# CHECK:      [Nr] Name      Type     Address  Off      Size   ES Flg Lk Inf Al
# CHECK:      ] .rela.plt RELA     [[#%x,]] [[#%x,]] 000018 18   A  1   0  8

# CHECK:      (PLTGOT)  0x0
# CHECK:      (PLTREL)  RELA

# NEG-NOT: ] .plt
# NEG-NOT: ] .iplt
# NEG-NOT: ] .got.plt

#--- a.s
  call foo
  call ifunc

.type ifunc, @gnu_indirect_function
.hidden ifunc
ifunc:
  ret

.data
.quad ifunc

#--- t
SECTIONS {
  .text : { *(.text) }
  /DISCARD/ : { *(.plt .iplt .got.plt) }
}
