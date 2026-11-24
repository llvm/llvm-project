# REQUIRES: aarch64

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t.o

# RUN: %lld -arch arm64 -lSystem -U _objc_msgSend -o %t.out %t.o
# RUN: llvm-nm %t.out | FileCheck %s
# RUN: %lld -arch arm64 -lSystem -U _objc_msgSend -dead_strip -o %t.out %t.o -map %t.map
# RUN: llvm-nm %t.out | FileCheck %s --check-prefix=DEAD \
# RUN:   --implicit-check-not=_foo --implicit-check-not='_objc_msgSend$deadsel'
# RUN: FileCheck %s --check-prefix=DEAD-MAP \
# RUN:   --implicit-check-not='_objc_msgSend$deadsel' < %t.map

# CHECK: _foo
# CHECK: _objc_msgSend$deadsel
# CHECK: _objc_msgSend$livesel

# DEAD: _objc_msgSend$livesel

# DEAD-MAP: 0x[[#%.8X,OBJC_STUBS:]] 0x00000020 __TEXT __objc_stubs
# DEAD-MAP-LABEL: # Symbols:
# DEAD-MAP: 0x[[#OBJC_STUBS]] 0x00000020 [  0] _objc_msgSend$livesel

.section __TEXT,__text

.globl _foo
_foo:
  bl  _objc_msgSend$deadsel
  ret

.globl _main
_main:
  bl  _objc_msgSend$livesel
  ret

.subsections_via_symbols
