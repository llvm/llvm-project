# REQUIRES: x86, zlib, zstd

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld -pie %t.o -o %t --compress-nonalloc-sections '*0=zlib' --compress-nonalloc-sections '*0=none'
# RUN: llvm-readelf -SrsX %t | FileCheck %s --check-prefix=CHECK1

# CHECK1:      Name       Type          Address     Off      Size     ES Flg Lk Inf Al
# CHECK1:      foo0       PROGBITS [[#%x,FOO0:]]    [[#%x,]] [[#%x,]] 00 A    0   0  8
# CHECK1-NEXT: foo1       PROGBITS [[#%x,FOO1:]]    [[#%x,]] [[#%x,]] 00 A    0   0  8
# CHECK1-NEXT: .text      PROGBITS [[#%x,TEXT:]]    [[#%x,]] [[#%x,]] 00 AX   0   0  4
# CHECK1:      nonalloc0  PROGBITS 0000000000000000 [[#%x,]] [[#%x,]] 00      0   0  8
# CHECK1-NEXT: nonalloc1  PROGBITS 0000000000000000 [[#%x,]] [[#%x,]] 00      0   0  8
# CHECK1-NEXT: .debug_str PROGBITS 0000000000000000 [[#%x,]] [[#%x,]] 01 MS   0   0  1

# CHECK1: 0000000000000010  0 NOTYPE  LOCAL  DEFAULT   [[#]] (nonalloc0) sym0
# CHECK1: 0000000000000008  0 NOTYPE  LOCAL  DEFAULT   [[#]] (nonalloc1) sym1

# RUN: ld.lld -pie %t.o -o %t2 --compress-nonalloc-sections '*0=zlib' --compress-nonalloc-sections .debug_str=zstd
# RUN: llvm-readelf -SrsX -x nonalloc0 -x .debug_str %t2 | FileCheck %s --check-prefix=CHECK2

# CHECK2:      Name       Type          Address     Off      Size     ES Flg Lk Inf Al
# CHECK2:      foo0       PROGBITS [[#%x,FOO0:]]    [[#%x,]] [[#%x,]] 00 A    0   0  8
# CHECK2-NEXT: foo1       PROGBITS [[#%x,FOO1:]]    [[#%x,]] [[#%x,]] 00 A    0   0  8
# CHECK2-NEXT: .text      PROGBITS [[#%x,TEXT:]]    [[#%x,]] [[#%x,]] 00 AX   0   0  4
# CHECK2:      nonalloc0  PROGBITS 0000000000000000 [[#%x,]] [[#%x,]] 00 C    0   0  1
# CHECK2-NEXT: nonalloc1  PROGBITS 0000000000000000 [[#%x,]] [[#%x,]] 00      0   0  8
# CHECK2-NEXT: .debug_str PROGBITS 0000000000000000 [[#%x,]] [[#%x,]] 01 MSC  0   0  1

# CHECK2: 0000000000000010  0 NOTYPE  LOCAL  DEFAULT   [[#]] (nonalloc0) sym0
# CHECK2: 0000000000000008  0 NOTYPE  LOCAL  DEFAULT   [[#]] (nonalloc1) sym1

# CHECK2:      Hex dump of section 'nonalloc0':
## zlib with ch_size=0x10
# CHECK2-NEXT: 01000000 00000000 10000000 00000000
# CHECK2-NEXT: 01000000 00000000 {{.*}}
# CHECK2:      Hex dump of section '.debug_str':
## zstd with ch_size=0x38
# CHECK2-NEXT: 02000000 00000000 38000000 00000000
# CHECK2-NEXT: 01000000 00000000 {{.*}}

# RUN: not ld.lld --compress-nonalloc-sections=foo %t.o -o /dev/null 2>&1 | \
# RUN:   FileCheck %s --check-prefix=ERR1 --implicit-check-not=error:
# ERR1:      error: --compress-nonalloc-sections: parse error, not 'section-glob=[none|zlib|zstd]'

# RUN: not ld.lld --compress-nonalloc-sections 'a[=zlib' %t.o -o /dev/null 2>&1 | \
# RUN:   FileCheck %s --check-prefix=ERR2 --implicit-check-not=error:
# ERR2:      error: --compress-nonalloc-sections: invalid glob pattern, unmatched '['

# RUN: not ld.lld %t.o -o /dev/null --compress-nonalloc-sections='.debug*=zlib-gabi' --compress-nonalloc-sections='.debug*=' 2>&1 | \
# RUN:   FileCheck -check-prefix=ERR3 %s
# ERR3:      unknown --compress-nonalloc-sections value: zlib-gabi
# ERR3-NEXT: --compress-nonalloc-sections: parse error, not 'section-glob=[none|zlib|zstd]'

.globl _start
_start:
  leaq __start_foo0(%rip), %rax
  leaq __stop_foo0(%rip), %rax
  ret

.section foo0,"a"
.balign 8
.quad .text-.
.quad .text-.
.section foo1,"a"
.balign 8
.quad .text-.
.quad .text-.
.section nonalloc0,""
.balign 8
.quad .text
.quad .text
sym0:
.section nonalloc1,""
.balign 8
.quad 42
sym1:

.section .debug_str,"MS",@progbits,1
.Linfo_string0:
  .asciz "AAAAAAAAAAAAAAAAAAAAAAAAAAA"
.Linfo_string1:
  .asciz "BBBBBBBBBBBBBBBBBBBBBBBBBBB"
