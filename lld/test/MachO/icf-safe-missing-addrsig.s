# REQUIRES: aarch64

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/with-addrsig.s -o %t/with-addrsig.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/without-addrsig.s -o %t/without-addrsig.o
# RUN: %lld -arch arm64 -lSystem --icf=safe -dylib -map %t/with-addrsig-safe.map -o %t/with-addrsig.dylib %t/with-addrsig.o
# RUN: %lld -arch arm64 -lSystem --icf=safe -dylib -map %t/without-addrsig-safe.map -o %t/without-addrsig.dylib %t/without-addrsig.o
# RUN: %lld -arch arm64 -lSystem --icf=safe_thunks -dylib -map %t/with-addrsig-safe-thunks.map -o %t/with-addrsig-thunks.dylib %t/with-addrsig.o
# RUN: %lld -arch arm64 -lSystem --icf=safe_thunks -dylib -map %t/without-addrsig-safe-thunks.map -o %t/without-addrsig-thunks.dylib %t/without-addrsig.o
# RUN: FileCheck %s --check-prefix=ADDRSIG-SAFE < %t/with-addrsig-safe.map
# RUN: FileCheck %s --check-prefix=NO-ADDRSIG-SAFE < %t/without-addrsig-safe.map
# RUN: FileCheck %s --check-prefix=ADDRSIG-SAFE-THUNKS < %t/with-addrsig-safe-thunks.map
# RUN: FileCheck %s --check-prefix=NO-ADDRSIG-SAFE-THUNKS < %t/without-addrsig-safe-thunks.map

## Input has addrsig section: _g1 and _g2 are address-significant, so _g2 is
## thunk-folded in safe_thunks ICF and remains untouched in safe ICF.
## _f2 is always body-folded into _f1 regardless of ICF level.

# ADDRSIG-SAFE:      0x00000008 [  2] _f1
# ADDRSIG-SAFE-NEXT: 0x00000000 [  2] _f2
# ADDRSIG-SAFE:      0x00000008 [  2] _g1
# ADDRSIG-SAFE-NEXT: 0x00000008 [  2] _g2

# ADDRSIG-SAFE-THUNKS:      0x00000008 [  2] _f1
# ADDRSIG-SAFE-THUNKS-NEXT: 0x00000000 [  2] _f2
# ADDRSIG-SAFE-THUNKS:      0x00000008 [  2] _g1
# ADDRSIG-SAFE-THUNKS:      0x00000004 [  2] _g2

## Input does not have addrsig section: everything is address-significant, so
## no folding happened in safe ICF, and _f2, _g2 are thunk-folded into _f1, _g1
## respectively.

# NO-ADDRSIG-SAFE:      0x00000008 [  2] _f1
# NO-ADDRSIG-SAFE-NEXT: 0x00000008 [  2] _f2
# NO-ADDRSIG-SAFE-NEXT: 0x00000008 [  2] _g1
# NO-ADDRSIG-SAFE-NEXT: 0x00000008 [  2] _g2

# NO-ADDRSIG-SAFE-THUNKS:      0x00000008 [  2] _f1
# NO-ADDRSIG-SAFE-THUNKS-NEXT: 0x00000008 [  2] _g1
# NO-ADDRSIG-SAFE-THUNKS:      0x00000004 [  2] _g2
# NO-ADDRSIG-SAFE-THUNKS-NEXT: 0x00000004 [  2] _f2

#--- with-addrsig.s
.subsections_via_symbols
.text
.p2align 2

.globl _f1
_f1:
  mov w0, #0
  ret

.globl _f2
_f2:
  mov w0, #0
  ret

.globl _g1
_g1:
  mov w0, #1
  ret

.globl _g2
_g2:
  mov w0, #1
  ret

.globl _call_all
_call_all:
  bl _f1
  bl _f2
  bl _g1
  bl _g2
  ret

.addrsig
.addrsig_sym _call_all
.addrsig_sym _g1
.addrsig_sym _g2

#--- without-addrsig.s
.subsections_via_symbols
.text
.p2align 2

.globl _f1
_f1:
  mov w0, #0
  ret

.globl _f2
_f2:
  mov w0, #0
  ret

.globl _g1
_g1:
  mov w0, #1
  ret

.globl _g2
_g2:
  mov w0, #1
  ret

.globl _call_all
_call_all:
  bl _f1
  bl _f2
  bl _g1
  bl _g2
  ret
