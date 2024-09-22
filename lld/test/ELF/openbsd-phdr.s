# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-openbsd randomdata.s -o randomdata.o
# RUN: ld.lld randomdata.o -o randomdata
# RUN: llvm-readelf -S -l randomdata | FileCheck %s --check-prefix=RANDOMDATA

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-openbsd /dev/null -o wxneeded.o
# RUN: ld.lld -z wxneeded wxneeded.o -o wxneeded
# RUN: llvm-readelf -l wxneeded | FileCheck %s --check-prefix=WXNEEDED

# RUN: ld.lld -T lds randomdata.o -o out
# RUN: llvm-readelf -S -l out | FileCheck %s --check-prefixes=RANDOMDATA,CHECK

# RANDOMDATA: Name                Type     Address            Off             Size   ES Flg Lk Inf Al
# RANDOMDATA: .openbsd.randomdata PROGBITS [[ADDR:[0-9a-f]+]] [[O:[0-9a-f]+]] 000008 00   A  0   0  1

# WXNEEDED:   Type              Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# WXNEEDED:   OPENBSD_WXNEEDED  0x000000 0x0000000000000000 0x0000000000000000 0x000000 0x000000 E   0

# RANDOMDATA: Type              Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# RANDOMDATA: OPENBSD_RANDOMIZE 0x[[O]]  0x[[ADDR]]         0x[[ADDR]]         0x000008 0x000008 R   0x1
# CHECK-NEXT: OPENBSD_BOOTDATA  0x000000 0x0000000000000000 0x0000000000000000 0x000000 0x000000 R   0
# CHECK-NEXT: OPENBSD_MUTABLE   0x000000 0x0000000000000000 0x0000000000000000 0x000000 0x000000 R   0
# CHECK-NEXT: OPENBSD_SYSCALLS  0x000000 0x0000000000000000 0x0000000000000000 0x000000 0x000000 R   0
# CHECK-NEXT: OPENBSD_WXNEEDED  0x000000 0x0000000000000000 0x0000000000000000 0x000000 0x000000 R   0

#--- randomdata.s
.section .openbsd.randomdata, "a"
.quad 0

#--- lds
PHDRS {
  text PT_LOAD FILEHDR PHDRS;
  rand PT_OPENBSD_RANDOMIZE;
  boot PT_OPENBSD_BOOTDATA;
  mutable PT_OPENBSD_MUTABLE;
  syscalls PT_OPENBSD_SYSCALLS;
  wxneeded PT_OPENBSD_WXNEEDED;
}
SECTIONS {
  . = SIZEOF_HEADERS;
  .text : { *(.text) }
  .openbsd.randomdata : { *(.openbsd.randomdata) } : rand
}
