# REQUIRES: x86
# RUN: split-file %s %t.dir && cd %t.dir

# RUN: llvm-mc -triple=x86_64-windows-cygnus -filetype=obj -o data-no-bss.obj data-no-bss.s
# RUN: lld-link -lldmingw -entry:main data-no-bss.obj -out:data-no-bss.exe
# RUN: llvm-objdump -s data-no-bss.exe | FileCheck --check-prefix=DATANOBSS %s

# RUN: llvm-mc -triple=x86_64-windows-cygnus -filetype=obj -o bss-no-data.obj bss-no-data.s
# RUN: lld-link -lldmingw -entry:main bss-no-data.obj -out:bss-no-data.exe
# RUN: llvm-objdump -s bss-no-data.exe | FileCheck --check-prefix=BSSNODATA %s

# RUN: llvm-mc -triple=x86_64-windows-cygnus -filetype=obj -o data-and-bss.obj data-and-bss.s
# RUN: lld-link -lldmingw -entry:main data-and-bss.obj -out:data-and-bss.exe
# RUN: llvm-objdump -s data-and-bss.exe | FileCheck --check-prefix=DATAANDBSS %s

#--- data-no-bss.s
.globl main
main:
  nop

.data
  .quad 1
  .byte 2

.section .data_cygwin_nocopy, "w"
  .align 4
  .quad 3
  .byte 4

.section .test, "w"
  .quad __data_start__
  .quad __data_end__
  .quad __bss_start__
  .quad __bss_end__

#--- bss-no-data.s
.globl main
main:
  nop

.bss
  .zero 8192

.section .test, "w"
  .quad __data_start__
  .quad __data_end__
  .quad __bss_start__
  .quad __bss_end__

#--- data-and-bss.s
.globl main
main:
  nop

.data
  .quad 1
  .byte 2

.section .data_cygwin_nocopy, "w"
  .align 4
  .quad 3
  .byte 4

.bss
  .zero 8192

.section .test, "w"
  .quad __data_start__
  .quad __data_end__
  .quad __bss_start__
  .quad __bss_end__

# DATANOBSS:      Contents of section .data:
# DATANOBSS-NEXT: 140003000 01000000 00000000 02000000 03000000
# DATANOBSS-NEXT: 140003010 00000000 04
# __data_start__ pointing at 0x140003000 and
# __data_end__   pointing at 0x140003009.
# DATANOBSS-NEXT: Contents of section .test:
# DATANOBSS-NEXT: 140004000 00300040 01000000 09300040 01000000
# DATANOBSS-NEXT: 140004010 18300040 01000000 18300040 01000000

# __bss_start__ pointing at 0x140003000 and
# __bss_end__   pointing at 0x140005000.
# BSSNODATA-NOT:  Contents of section .data:
# BSSNODATA:      Contents of section .test:
# BSSNODATA-NEXT: 140005000 00300040 01000000 00300040 01000000
# BSSNODATA-NEXT: 140005010 00300040 01000000 00500040 01000000

# DATAANDBSS:      Contents of section .data:
# DATAANDBSS-NEXT: 140003000 01000000 00000000 02000000 03000000
# DATAANDBSS-NEXT: 140003010 00000000 04000000 00000000 00000000
# __data_start__ pointing at 0x140003000 and
# __data_end__   pointing at 0x140003009.
# __bss_start__ pointing at 0x140003018 and
# __bss_end__   pointing at 0x140005018.
# DATAANDBSS:      1400031f0 00000000 00000000 00000000 00000000
# DATAANDBSS-NEXT: Contents of section .test:
# DATAANDBSS-NEXT: 140006000 00300040 01000000 09300040 01000000
# DATAANDBSS-NEXT: 140006010 18300040 01000000 18500040 01000000
