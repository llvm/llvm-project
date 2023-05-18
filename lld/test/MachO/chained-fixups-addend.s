# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/dylib.s -o %t/dylib.o
# RUN: %lld -lSystem -dylib %t/dylib.o -o %t/libdylib.dylib

## FileCheck does not like wrapping arithmetic, so we specify all 3 check variables manually:
##   ADDEND  := inline/outline addend, unsigned
##   OUTLINE := outline addend, signed
##   REBASE  := target of rebase; 0x1000 + ADDEND, unsigned

## We can use the DYLD_CHAINED_IMPORT import format if 0 <= ADDEND <= 255 bytes.
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/main.s -o %t/main.o --defsym ADDEND=0
# RUN: %lld -lSystem -dylib %t/main.o -L%t -ldylib -fixup_chains -o %t/out
# RUN: llvm-objdump --macho --chained-fixups --dyld-info %t/out | \
# RUN:     FileCheck %s -D#OUTLINE=0 -D#ADDEND=0 -D#%x,REBASE=0x1000 --check-prefixes=IMPORT,COMMON
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/main.s -o %t/main.o --defsym ADDEND=255
# RUN: %lld -lSystem -dylib %t/main.o -L%t -ldylib -fixup_chains -o %t/out
# RUN: llvm-objdump --macho --chained-fixups --dyld-info %t/out | \
# RUN:     FileCheck %s -D#OUTLINE=0 -D#ADDEND=255 -D#%x,REBASE=0x10FF --check-prefixes=IMPORT,COMMON

## DYLD_CHAINED_IMPORT_ADDEND is used if the addend fits in a 32-bit signed integer.
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/main.s -o %t/main.o --defsym ADDEND=-1
# RUN: %lld -lSystem -dylib %t/main.o -L%t -ldylib -fixup_chains -o %t/out
# RUN: llvm-objdump --macho --chained-fixups --dyld-info %t/out | \
# RUN:     FileCheck %s -D#%d,OUTLINE=-1 -D#%x,ADDEND=0xFFFFFFFFFFFFFFFF -D#%x,REBASE=0xFFF \
# RUN:     --check-prefixes=IMPORT-ADDEND,COMMON
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/main.s -o %t/main.o --defsym ADDEND=256
# RUN: %lld -lSystem -dylib %t/main.o -L%t -ldylib -fixup_chains -o %t/out
# RUN: llvm-objdump --macho --chained-fixups --dyld-info %t/out | \
# RUN:     FileCheck %s -D#OUTLINE=256 -D#ADDEND=256 -D#%x,REBASE=0x1100 \
# RUN:     --check-prefixes=IMPORT-ADDEND,COMMON

## Otherwise, DYLD_CHAINED_IMPORT_ADDEND64 is used.
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/main.s -o %t/main.o --defsym ADDEND=0x100000000
# RUN: %lld -lSystem -dylib %t/main.o -L%t -ldylib -fixup_chains -o %t/out
# RUN: llvm-objdump --macho --chained-fixups --dyld-info %t/out | \
# RUN:     FileCheck %s -D#%x,OUTLINE=0x100000000 -D#%x,ADDEND=0x100000000 \
# RUN:     -D#%x,REBASE=0x100001000 --check-prefixes=IMPORT-ADDEND64,COMMON

# COMMON:      dyld information:
# COMMON-NEXT: segment  section address pointer  type  addend            dylib     symbol/vm address
# COMMON-NEXT: __DATA   __data    {{.*}}         bind  0x[[#%X, ADDEND]] libdylib  _dysym
# COMMON-NEXT: __DATA   __data    {{.*}}         rebase                            0x[[#%X, REBASE]]

# COMMON:        chained fixups header (LC_DYLD_CHAINED_FIXUPS)
# IMPORT:          imports_format = 1 (DYLD_CHAINED_IMPORT)
# IMPORT-ADDEND:   imports_format = 2 (DYLD_CHAINED_IMPORT_ADDEND)
# IMPORT-ADDEND64: imports_format = 3 (DYLD_CHAINED_IMPORT_ADDEND64)

# IMPORT:             dyld chained import[0]
# IMPORT-ADDEND:      dyld chained import addend[0]
# IMPORT-ADDEND64:    dyld chained import addend64[0]
# COMMON-NEXT:          lib_ordinal = 2 (libdylib)
# COMMON-NEXT:          weak_import = 0
# COMMON-NEXT:          name_offset = 0 (_dysym)
# IMPORT-ADDEND-NEXT:   addend      = [[#%d, OUTLINE]]
# IMPORT-ADDEND64-NEXT: addend      = [[#%d, OUTLINE]]

#--- dylib.s
.globl _dysym

_dysym:
  ret

#--- main.s
.globl _dysym, _local

.data
_local:
.skip 128

.p2align 3
.quad _dysym + ADDEND
.quad _local + ADDEND
