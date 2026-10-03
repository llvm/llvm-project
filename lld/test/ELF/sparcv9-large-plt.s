# REQUIRES: sparc
# RUN: %python %S/Inputs/sparcv9-large-plt.py 32766 > %t.s
# RUN: llvm-mc --position-independent -filetype=obj -triple=sparcv9 %t.s -o %t.o
# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-readelf -SW -r %t.so | FileCheck %s

# CHECK-NOT: .got.plt
# CHECK:     .plt PROGBITS [[#%x,PLT:]] {{[0-9a-f]+}} 100040
# CHECK-NOT: .got.plt
# CHECK:     [[#%x,PLT + 0x100030]] {{.*}} R_SPARC_JMP_SLOT {{.*}} foo32764
# CHECK:     [[#%x,PLT + 0x100038]] {{.*}} R_SPARC_JMP_SLOT {{.*}} foo32765
