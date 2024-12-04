# REQUIRES: aarch64
## __rela_iplt_start and __rela_iplt_end must surround the IRELATIVE relocation
## list even if moved to .rel[a].plt because of packed relocation sections.

# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-android %s -o %t.o
# RUN: ld.lld --pack-dyn-relocs=android %t.o -o %t
# RUN: llvm-readelf -sS %t | FileCheck %s

# CHECK: .rela.plt         RELA            0000000000200158 000158 000018 18  AI  0   5  8
# CHECK: 0000000000200158     0 NOTYPE  LOCAL  HIDDEN      1 __rela_iplt_start
# CHECK: 0000000000200170     0 NOTYPE  LOCAL  HIDDEN      1 __rela_iplt_end

.text
.type foo, %gnu_indirect_function
.globl foo
foo:
  ret

.globl _start
_start:
  bl foo

.data
.balign 8
.quad __rela_iplt_start
.quad __rela_iplt_end
