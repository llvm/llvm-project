# REQUIRES: aarch64
## On arm64_32, GOT and lazy-pointer entries are 4 bytes. Writing them as
## 8-byte values clobbers the first word of the following section (racily,
## depending on the order in which sections are written).
# RUN: llvm-mc -filetype=obj -triple=arm64_32-apple-watchos5 %s -o %t.o
# RUN: %no-arg-lld -dylib -arch arm64_32 -platform_version watchos 5 5 \
# RUN:   -undefined dynamic_lookup %t.o -o %t.dylib
# RUN: llvm-objdump --macho --section-headers %t.dylib | FileCheck %s --check-prefix=SECTIONS
# RUN: llvm-objdump --macho --full-contents --section=__la_symbol_ptr \
# RUN:   --section=__got --section=__data --section=__const %t.dylib | \
# RUN:   FileCheck %s --check-prefix=CONTENTS

## __la_symbol_ptr is immediately followed by __data.
# SECTIONS:      __got           00000008 0000[[#%x,GOT:]] DATA
# SECTIONS-NEXT: __la_symbol_ptr 00000004 0000[[#%x,GOT+8]] DATA
# SECTIONS-NEXT: __data          00000008 0000[[#%x,GOT+12]] DATA

# CONTENTS:      Contents of section __DATA_CONST,__const:
# CONTENTS-NEXT: 44332211
# CONTENTS:      Contents of section __DATA,__got:
# CONTENTS-NEXT: 00000000 00000000
# CONTENTS:      Contents of section __DATA,__la_symbol_ptr:
# CONTENTS-NEXT: {{[0-9a-f]{8} }}
## The first word of __data survives the 4-byte lazy pointer before it.
# CONTENTS:      Contents of section __DATA,__data:
# CONTENTS-NEXT: 88776655 00000000

.text
.globl _main
.p2align 2
_main:
  bl _lazy_function
  adrp x0, _got_symbol@GOTPAGE
  ldr w0, [x0, _got_symbol@GOTPAGEOFF]
  ret

.section __DATA_CONST,__const
.p2align 2
_const_word:
  .long 0x11223344

.section __DATA,__data
.p2align 2
_data_word:
  .long 0x55667788
