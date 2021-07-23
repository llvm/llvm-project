# REQUIRES: x86

# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/test.o
# RUN: %lld -execute -o %t/exec %t/test.o -lSystem
# RUN: %lld -dylib -o %t/dylib %t/test.o -lSystem
# RUN: %lld -bundle -o %t/bundle %t/test.o -lSystem

# RUN: llvm-objdump --macho --all-headers %t/exec | \
# RUN:     FileCheck %s --check-prefixes=EXEC,COMMON
# RUN: llvm-objdump --macho --all-headers %t/dylib | \
# RUN:     FileCheck %s --check-prefixes=DYLIB,COMMON
# RUN: llvm-objdump --macho --all-headers %t/bundle | \
# RUN:     FileCheck %s --check-prefix=COMMON

## Check that load commands and sections within segments occur in the proper
## sequence. On ARM64 kernel is especially picky about layout, and will
## barf with errno=EBADMACHO when the sequence is wrong.

# EXEC: cmd LC_SEGMENT_64
# EXEC: segname __PAGEZERO

# COMMON: cmd LC_SEGMENT_64
# COMMON: segname __TEXT
# COMMON: sectname __text
# COMMON: segname __TEXT
# COMMON: sectname __cstring
# COMMON: segname __TEXT
# COMMON: cmd LC_SEGMENT_64
# COMMON: segname __DATA_CONST
# COMMON: sectname __got
# COMMON: segname __DATA_CONST
# COMMON: sectname __const
# COMMON: segname __DATA_CONST
# COMMON: cmd LC_SEGMENT_64
# COMMON: segname __DATA
# COMMON: sectname __data
# COMMON: segname __DATA
# COMMON: cmd LC_SEGMENT_64
# COMMON: segname __LINKEDIT
# COMMON: cmd LC_DYLD_INFO_ONLY
# COMMON: cmd LC_SYMTAB
# COMMON: cmd LC_DYSYMTAB

# EXEC: cmd LC_LOAD_DYLINKER
# DYLIB: cmd LC_ID_DYLIB

# COMMON: cmd LC_UUID
# COMMON: cmd LC_BUILD_VERSION

# EXEC: cmd LC_MAIN

# COMMON: cmd LC_LOAD_DYLIB

.section __TEXT,__cstring
_str:
  .asciz "Help me! I'm trapped in a test!\n"

.globl _mutable
.section __DATA,__data
mutable:
  .long 0x1234

.globl _constant
.section __DATA,__const
constant:
  .long 0x4567

.text
.global _main
_main:
  mov ___nan@GOTPCREL(%rip), %rax
  ret
