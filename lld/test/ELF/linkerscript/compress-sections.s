# REQUIRES: x86, zlib

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: ld.lld -T a.lds a.o --compress-sections nonalloc=zlib --compress-sections str=zlib -o out
# RUN: llvm-readelf -SsXz -p str out | FileCheck %s

# CHECK:      Name     Type            Address   Off      Size     ES Flg  Lk Inf Al
# CHECK:      nonalloc PROGBITS 0000000000000000 [[#%x,]] [[#%x,]] 00   C   0   0  1
# CHECK-NEXT: str      PROGBITS 0000000000000000 [[#%x,]] [[#%x,]] 01 MSC   0   0  1

# CHECK:      0000000000000000  0 NOTYPE  GLOBAL DEFAULT [[#]] (nonalloc) nonalloc_start
# CHECK:      0000000000000023  0 NOTYPE  GLOBAL DEFAULT [[#]] (nonalloc) nonalloc_end
# CHECK:      String dump of section 'str':
# CHECK-NEXT: [     0] AAA
# CHECK-NEXT: [     4] BBB

## TODO The uncompressed size of 'nonalloc' is dependent on linker script
## commands, which is not handled. We should report an error.
# RUN: ld.lld -T b.lds a.o --compress-sections nonalloc=zlib

#--- a.s
.globl _start
_start:
  ret

.section nonalloc0,""
.balign 8
.quad .text
.quad .text
.section nonalloc1,""
.balign 8
.quad 42

.section str,"MS",@progbits,1
  .asciz "AAA"
  .asciz "BBB"

#--- a.lds
SECTIONS {
  .text : { *(.text) }
  c = SIZEOF(.text);
  b = c+1;
  a = b+1;
  nonalloc : {
    nonalloc_start = .;
## In general, using data commands is error-prone. This case is correct, though.
    *(nonalloc*) QUAD(SIZEOF(.text))
    . += a;
    nonalloc_end = .;
  }
  str : { *(str) }
}

#--- b.lds
SECTIONS {
  nonalloc : { *(nonalloc*) . += a; }
  .text : { *(.text) }
  a = b+1;
  b = c+1;
  c = SIZEOF(.text);
}
