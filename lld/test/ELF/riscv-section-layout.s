# REQUIRES: riscv
## Test RISC-V specific section layout. See also section-layout.s and riscv-gp.s.

# RUN: llvm-mc -filetype=obj -triple=riscv32 %s -o %t.32.o
# RUN: ld.lld -pie %t.32.o -o %t.32
# RUN: llvm-readelf -S -sX %t.32 | FileCheck %s --check-prefix=NOSDATA
# RUN: llvm-mc -filetype=obj -triple=riscv32 --defsym=SDATA=1 %s -o %t.32s.o
# RUN: ld.lld -pie %t.32s.o -o %t.32s
# RUN: llvm-readelf -S -sX %t.32s | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=riscv64 %s -o %t.64.o
# RUN: ld.lld -pie %t.64.o -o %t.64
# RUN: llvm-readelf -S -sX %t.64 | FileCheck %s --check-prefix=NOSDATA
# RUN: llvm-mc -filetype=obj -triple=riscv64 --defsym=SDATA=1 %s -o %t.64s.o
# RUN: ld.lld -pie %t.64s.o -o %t.64s
# RUN: llvm-readelf -S -sX %t.64s | FileCheck %s

# NOSDATA:      .rodata  PROGBITS
# NOSDATA-NEXT: .text    PROGBITS
# NOSDATA-NEXT: .tdata   PROGBITS [[#%x,TDATA:]]
# NOSDATA-NEXT: .tbss    NOBITS
# NOSDATA-NEXT: .dynamic DYNAMIC
# NOSDATA-NEXT: .got     PROGBITS
# NOSDATA-NEXT: .relro_padding NOBITS
# NOSDATA-NEXT: .data    PROGBITS [[#%x,DATA:]]
# NOSDATA-NEXT: .bss     NOBITS   [[#%x,BSS:]]

## If there is an undefined reference to __global_pointer$ but .sdata doesn't
## exist, define __global_pointer$ and set its st_shndx arbitrarily to 1.
## The symbol value should not be used by the program.

# NOSDATA-DAG:  [[#]]: {{.*}}                 0 NOTYPE  GLOBAL DEFAULT [[#]] (.text) _etext
# NOSDATA-DAG:  [[#]]: {{0*}}[[#BSS]]         0 NOTYPE  GLOBAL DEFAULT [[#]] (.data) _edata
# NOSDATA-DAG:  [[#]]: {{0*}}[[#BSS]]         0 NOTYPE  GLOBAL DEFAULT [[#]] (.bss) __bss_start
# NOSDATA-DAG:  [[#]]: {{0*}}800              0 NOTYPE  GLOBAL DEFAULT  1 (.dynsym) __global_pointer$

# CHECK:      .rodata    PROGBITS
# CHECK-NEXT: .srodata   PROGBITS
# CHECK-NEXT: .text      PROGBITS
# CHECK-NEXT: .tdata     PROGBITS
# CHECK-NEXT: .tbss      NOBITS
# CHECK-NEXT: .dynamic   DYNAMIC
# CHECK-NEXT: .got       PROGBITS
# CHECK-NEXT: .relro_padding NOBITS
# CHECK-NEXT: .data      PROGBITS
# CHECK-NEXT: .sdata     PROGBITS [[#%x,SDATA:]]
# CHECK-NEXT: .sbss      NOBITS   [[#%x,SBSS:]]
# CHECK-NEXT: .bss       NOBITS

# CHECK-DAG:  [[#]]: {{0*}}[[#SBSS]]        0 NOTYPE  GLOBAL DEFAULT [[#]] (.sdata) _edata
# CHECK-DAG:  [[#]]: {{0*}}[[#SBSS]]        0 NOTYPE  GLOBAL DEFAULT [[#]] (.sbss) __bss_start
# CHECK-DAG:  [[#]]: {{0*}}[[#SDATA+0x800]] 0 NOTYPE  GLOBAL DEFAULT [[#]] (.sdata) __global_pointer$

.globl _etext, _edata, __bss_start
  lla gp, __global_pointer$

.section .rodata,"a",@progbits; .space 1
.section .data,"aw",@progbits; .long _GLOBAL_OFFSET_TABLE_ - .
.section .bss,"aw",@nobits; .space 1
.section .tdata,"awT",@progbits; .space 1
.section .tbss,"awT",@nobits; .space 1
.ifdef SDATA
.section .sdata,"aw",@progbits; .space 1
.section .sdata.suffix,"aw",@progbits; .space 1
.section .sbss,"aw",@nobits; .space 1
.section .sbss.suffix,"aw",@nobits; .space 1
.section .srodata,"a",@progbits; .space 1
.section .srodata.suffix,"a",@progbits; .space 1
.endif
