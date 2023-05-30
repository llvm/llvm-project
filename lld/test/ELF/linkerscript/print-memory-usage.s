# REQUIRES: x86

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/a1.s -o %t/a1.o
# RUN: ld.lld -T %t/1.t %t/a1.o -o %t/a1 --print-memory-usage \
# RUN:     | FileCheck %s --check-prefix=CHECK1 --match-full-lines --strict-whitespace
# RUN: ld.lld -T %t/2.t %t/a1.o -o %t/a2 --print-memory-usage \
# RUN:     | FileCheck %s --check-prefix=CHECK2 --match-full-lines --strict-whitespace
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/a2.s -o %t/a2.o
# RUN: ld.lld -T %t/3.t %t/a2.o -o %t/a3 --print-memory-usage \
# RUN:     | FileCheck %s --check-prefix=CHECK3 --match-full-lines --strict-whitespace

#      CHECK1:Memory region         Used Size  Region Size  %age Used
# CHECK1-NEXT:             ROM:           4 B         1 KB      0.39%
# CHECK1-NEXT:             RAM:           4 B       256 KB      0.00%
#  CHECK1-NOT:{{.}}

# CHECK2:Memory region         Used Size  Region Size  %age Used
# CHECK2-NOT:{{.}}

#      CHECK3:Memory region         Used Size  Region Size  %age Used
# CHECK3-NEXT:             ROM:        256 KB         1 MB     25.00%
# CHECK3-NEXT:             RAM:          32 B         2 GB      0.00%
#  CHECK3-NOT:{{.}}

#--- a1.s
.text
.globl _start
_start:
  .long 1

.data
.globl b
b:
  .long 2

#--- a2.s
.text
.globl _start
_start:
  .space 256*1024

.data
.globl b
b:
  .space 32

#--- 1.t
MEMORY {
  ROM (RX) : ORIGIN = 0x0, LENGTH = 1K
  RAM (W)  : ORIGIN = 0x100000, LENGTH = 256K
}
SECTIONS {
  . = 0;
  .text : { *(.text) }
  .data : { *(.data) }
}

#--- 2.t
SECTIONS {
  . = 0;
  .text : { *(.text) }
  .data : { *(.data) }
}

#--- 3.t
MEMORY {
  ROM (RX) : ORIGIN = 0x0, LENGTH = 1M
  RAM (W)  : ORIGIN = 0x1000000, LENGTH = 2048M
}
SECTIONS {
  . = 0;
  .text : { *(.text) }
  .data : { *(.data) }
}
