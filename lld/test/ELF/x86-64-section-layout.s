# REQUIRES: x86
## Test the placement of .lrodata, .lbss, .ldata, and their -fdata-sections variants.
## See also section-layout.s.

# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=x86_64 --defsym=BSS=1 a.s -o a.o
# RUN: ld.lld --section-start=.note=0x200300 a.o -o a
# RUN: llvm-readelf -S -l -sX a | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a1.o
# RUN: ld.lld --section-start=.note=0x200300 a1.o -o a1
# RUN: llvm-readelf -S -sX a1 | FileCheck %s --check-prefix=CHECK1

# RUN: ld.lld -T b.lds -z norelro a.o -z lrodata-after-bss -z nolrodata-after-bss -o b --fatal-warnings
# RUN: llvm-readelf -S -l b | FileCheck %s --check-prefix=CHECK2

# RUN: ld.lld --section-start=.note=0x200300 a.o -z lrodata-after-bss -o a3
# RUN: llvm-readelf -S -l -sX a3 | FileCheck %s --check-prefix=CHECK3

# CHECK:       Name              Type            Address          Off    Size   ES Flg Lk Inf Al
# CHECK-NEXT:                    NULL            0000000000000000 000000 000000 00      0   0  0
# CHECK-NEXT:  .note             NOTE            0000000000200300 000300 000001 00   A  0   0  1
# CHECK-NEXT:  .lrodata          PROGBITS        0000000000200301 000301 000002 00  Al  0   0  1
# CHECK-NEXT:  .rodata           PROGBITS        0000000000200303 000303 000001 00   A  0   0  1
# CHECK-NEXT:  .text             PROGBITS        0000000000201304 000304 000001 00  AX  0   0  4
# CHECK-NEXT:  .tdata            PROGBITS        0000000000202305 000305 000001 00 WAT  0   0  1
# CHECK-NEXT:  .tbss             NOBITS          0000000000202306 000306 000002 00 WAT  0   0  1
# CHECK-NEXT:  .relro_padding    NOBITS          0000000000202306 000306 000cfa 00  WA  0   0  1
# CHECK-NEXT:  .data             PROGBITS        0000000000203306 000306 000001 00  WA  0   0  1
# CHECK-NEXT:  .bss              NOBITS          0000000000203307 000307 001800 00  WA  0   0  1
## We spend size(.bss) % MAXPAGESIZE bytes for .bss.
# CHECK-NEXT:  .ldata            PROGBITS        0000000000205b07 000b07 000002 00 WAl  0   0  1
# CHECK-NEXT:  .ldata2           PROGBITS        0000000000205b09 000b09 000001 00 WAl  0   0  1
# CHECK-NEXT:  .lbss             NOBITS          0000000000205b0a 000b0a 001201 00 WAl  0   0  1
# CHECK-NEXT:  .comment          PROGBITS        0000000000000000 000b0a {{.*}} 01  MS  0   0  1

# CHECK:       Program Headers:
# CHECK-NEXT:    Type  Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# CHECK-NEXT:    PHDR  0x000040 0x0000000000200040 0x0000000000200040 {{.*}}   {{.*}}   R   0x8
# CHECK-NEXT:    LOAD  0x000000 0x0000000000200000 0x0000000000200000 0x000304 0x000304 R   0x1000
# CHECK-NEXT:    LOAD  0x000304 0x0000000000201304 0x0000000000201304 0x000001 0x000001 R E 0x1000
# CHECK-NEXT:    LOAD  0x000305 0x0000000000202305 0x0000000000202305 0x000001 0x000cfb RW  0x1000
# CHECK-NEXT:    LOAD  0x000306 0x0000000000203306 0x0000000000203306 0x000001 0x001801 RW  0x1000
# CHECK-NEXT:    LOAD  0x000b07 0x0000000000205b07 0x0000000000205b07 0x000003 0x001204 RW  0x1000

# CHECK:       0000000000201304     0 NOTYPE  GLOBAL DEFAULT [[#]] (.text)   _start
# CHECK-NEXT:  0000000000201305     0 NOTYPE  GLOBAL DEFAULT [[#]] (.text)   _etext
# CHECK-NEXT:  0000000000203307     0 NOTYPE  GLOBAL DEFAULT [[#]] (.data)   _edata
# CHECK-NEXT:  0000000000206d0b     0 NOTYPE  GLOBAL DEFAULT [[#]] (.lbss)   _end

# CHECK1:      .data      PROGBITS        0000000000203306 000306 000001 00  WA  0   0  1
# CHECK1-NEXT: .ldata     PROGBITS        0000000000203307 000307 000002 00 WAl  0   0  1
# CHECK1-NEXT: .ldata2    PROGBITS        0000000000203309 000309 000001 00 WAl  0   0  1
# CHECK1-NEXT: .comment   PROGBITS        0000000000000000 00030a {{.*}} 01  MS  0   0  1

# CHECK1:       0000000000201304     0 NOTYPE  GLOBAL DEFAULT [[#]] (.text)   _start
# CHECK1-NEXT:  0000000000201305     0 NOTYPE  GLOBAL DEFAULT [[#]] (.text)   _etext
# CHECK1-NEXT:  0000000000203307     0 NOTYPE  GLOBAL DEFAULT [[#]] (.data)   _edata
# CHECK1-NEXT:  000000000020330a     0 NOTYPE  GLOBAL DEFAULT [[#]] (.ldata2) _end

# CHECK2:      .note      NOTE            0000000000200300 000300 000001 00   A  0   0  1
# CHECK2-NEXT: .lrodata   PROGBITS        0000000000200301 000301 000001 00  Al  0   0  1
## With a SECTIONS command, we suppress the default rule placing .lrodata.* into .lrodata.
# CHECK2-NEXT: .lrodata.1 PROGBITS        0000000000200302 000302 000001 00  Al  0   0  1
# CHECK2-NEXT: .rodata    PROGBITS        0000000000200303 000303 000001 00   A  0   0  1
# CHECK2-NEXT: .text      PROGBITS        0000000000200304 000304 000001 00  AX  0   0  4
# CHECK2-NEXT: .tdata     PROGBITS        0000000000200305 000305 000001 00 WAT  0   0  1
# CHECK2-NEXT: .tbss      NOBITS          0000000000200306 000306 000001 00 WAT  0   0  1
# CHECK2-NEXT: .tbss.1    NOBITS          0000000000200307 000306 000001 00 WAT  0   0  1
# CHECK2-NEXT: .data      PROGBITS        0000000000200306 000306 000001 00  WA  0   0  1
# CHECK2-NEXT: .bss       NOBITS          0000000000200307 000307 001800 00  WA  0   0  1
# CHECK2-NEXT: .ldata     PROGBITS        0000000000201b07 001b07 000002 00 WAl  0   0  1
# CHECK2-NEXT: .ldata2    PROGBITS        0000000000201b09 001b09 000001 00 WAl  0   0  1
# CHECK2-NEXT: .lbss      NOBITS          0000000000201b0a 001b0a 001201 00 WAl  0   0  1
# CHECK2-NEXT: .comment   PROGBITS        0000000000000000 001b0a {{.*}} 01  MS  0   0  1

# CHECK2:      Program Headers:
# CHECK2-NEXT:   Type  Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# CHECK2-NEXT:   PHDR  0x000040 0x0000000000200040 0x0000000000200040 {{.*}}   {{.*}}   R   0x8
# CHECK2-NEXT:   LOAD  0x000000 0x0000000000200000 0x0000000000200000 0x000304 0x000304 R   0x1000
# CHECK2-NEXT:   LOAD  0x000304 0x0000000000200304 0x0000000000200304 0x000001 0x000001 R E 0x1000
# CHECK2-NEXT:   LOAD  0x000305 0x0000000000200305 0x0000000000200305 0x001805 0x002a06 RW  0x1000
# CHECK2-NEXT:   TLS   0x000305 0x0000000000200305 0x0000000000200305 0x000001 0x000003 R   0x1

# CHECK3:       Name              Type            Address          Off    Size   ES Flg Lk Inf Al
# CHECK3-NEXT:                    NULL            0000000000000000 000000 000000 00      0   0  0
# CHECK3-NEXT:  .note             NOTE            0000000000200300 000300 000001 00   A  0   0  1
# CHECK3-NEXT:  .rodata           PROGBITS        0000000000200301 000301 000001 00   A  0   0  1
# CHECK3-NEXT:  .text             PROGBITS        0000000000201304 000304 000001 00  AX  0   0  4
# CHECK3-NEXT:  .tdata            PROGBITS        0000000000202305 000305 000001 00 WAT  0   0  1
# CHECK3-NEXT:  .tbss             NOBITS          0000000000202306 000306 000002 00 WAT  0   0  1
# CHECK3-NEXT:  .relro_padding    NOBITS          0000000000202306 000306 000cfa 00  WA  0   0  1
# CHECK3-NEXT:  .data             PROGBITS        0000000000203306 000306 000001 00  WA  0   0  1
# CHECK3-NEXT:  .bss              NOBITS          0000000000203307 000307 001800 00  WA  0   0  1
## We spend (size(.bss) + size(.lbss)) % MAXPAGESIZE bytes.
# CHECK3-NEXT:  .lbss             NOBITS          0000000000204b07 000307 001201 00 WAl  0   0  1
# CHECK3-NEXT:  .lrodata          PROGBITS        0000000000206d08 000d08 000002 00  Al  0   0  1
# CHECK3-NEXT:  .ldata            PROGBITS        0000000000207d0a 000d0a 000002 00 WAl  0   0  1
# CHECK3-NEXT:  .ldata2           PROGBITS        0000000000207d0c 000d0c 000001 00 WAl  0   0  1
# CHECK3-NEXT:  .comment          PROGBITS        0000000000000000 000d0d {{.*}} 01  MS  0   0  1

# CHECK3:       Program Headers:
# CHECK3-NEXT:    Type  Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# CHECK3-NEXT:    PHDR  0x000040 0x0000000000200040 0x0000000000200040 {{.*}}   {{.*}}   R   0x8
# CHECK3-NEXT:    LOAD  0x000000 0x0000000000200000 0x0000000000200000 0x000302 0x000302 R   0x1000
# CHECK3-NEXT:    LOAD  0x000304 0x0000000000201304 0x0000000000201304 0x000001 0x000001 R E 0x1000
# CHECK3-NEXT:    LOAD  0x000305 0x0000000000202305 0x0000000000202305 0x000001 0x000cfb RW  0x1000
# CHECK3-NEXT:    LOAD  0x000306 0x0000000000203306 0x0000000000203306 0x000001 0x002a02 RW  0x1000
# CHECK3-NEXT:    LOAD  0x000d08 0x0000000000206d08 0x0000000000206d08 0x000002 0x000002 R   0x1000
# CHECK3-NEXT:    LOAD  0x000d0a 0x0000000000207d0a 0x0000000000207d0a 0x000003 0x000003 RW  0x1000
# CHECK3-NEXT:    TLS   0x000305 0x0000000000202305 0x0000000000202305 0x000001 0x000003 R   0x1

# CHECK3:       0000000000201304     0 NOTYPE  GLOBAL DEFAULT [[#]] (.text)   _start
# CHECK3-NEXT:  0000000000201305     0 NOTYPE  GLOBAL DEFAULT [[#]] (.text)   _etext
# CHECK3-NEXT:  0000000000203307     0 NOTYPE  GLOBAL DEFAULT [[#]] (.data)   _edata
# CHECK3-NEXT:  0000000000207d0d     0 NOTYPE  GLOBAL DEFAULT [[#]] (.ldata2) _end

#--- a.s
.globl _start, _etext, _edata, _end
_start:
  ret

.section .note,"a",@note; .space 1
.section .rodata,"a",@progbits; .space 1
.section .data,"aw",@progbits; .space 1
.ifdef BSS
## .bss is large than one MAXPAGESIZE to test file offsets.
.section .bss,"aw",@nobits; .space 0x1800
.endif
.section .tdata,"awT",@progbits; .space 1
.section .tbss,"awT",@nobits; .space 1
.section .tbss.1,"awT",@nobits; .space 1

.section .lrodata,"al"; .space 1
.section .lrodata.1,"al"; .space 1
.section .ldata,"awl"; .space 1
## Input .ldata.rel.ro sections are placed in the output .ldata section.
.section .ldata.rel.ro,"awl"; .space 1
.ifdef BSS
.section .lbss,"awl",@nobits; .space 0x1200
## Input .lbss.rel.ro sections are placed in the output .lbss section.
.section .lbss.rel.ro,"awl",@nobits; .space 1
.endif
.section .ldata2,"awl"; .space 1

#--- b.lds
SECTIONS {
  . = 0x200300;
  .rodata : {}
  .text : {}
  .data : {}
  .bss : {}
  .ldata : { *(.ldata .ldata.*) }
  .ldata2 : {}
  .lbss : { *(.lbss .lbss.*) }
}
