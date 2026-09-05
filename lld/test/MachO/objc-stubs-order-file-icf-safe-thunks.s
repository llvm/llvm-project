# REQUIRES: aarch64

## An order-file symbol folded to a safe thunk should still order the ObjC stub
## called by the original folded body.

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/input.s -o %t/input.o
# RUN: %lld -arch arm64 -e _entry -U _objc_msgSend -o %t/out %t/input.o \
# RUN:   -objc_stubs_small --icf=safe_thunks -order_file %t/order
# RUN: llvm-objdump --no-show-raw-insn -d %t/out | \
# RUN:   FileCheck %s --check-prefix=THUNK
# RUN: llvm-objdump --no-show-raw-insn --section=__TEXT,__objc_stubs --macho \
# RUN:   %t/out | FileCheck %s --check-prefix=ORDERED

# THUNK-LABEL: <_dup>:
# THUNK-NEXT:  b 0x{{[0-9a-f]+}} <_hot>

# ORDERED:      Contents of (__TEXT,__objc_stubs) section
# ORDERED-NEXT: _objc_msgSend$hot:
# ORDERED:      _objc_msgSend$cold:
# ORDERED:      _objc_msgSend$mild:

#--- input.s
.text
.globl _cold
_cold:
  bl _objc_msgSend$cold
  ret

.globl _mild
_mild:
  bl _objc_msgSend$mild
  ret

.globl _hot
_hot:
  bl _objc_msgSend$hot
  ret

.globl _dup
_dup:
  bl _objc_msgSend$hot
  ret

.globl _entry
_entry:
  bl _cold
  bl _mild
  bl _hot
  bl _dup
  ret

.addrsig
.addrsig_sym _hot
.addrsig_sym _dup

.subsections_via_symbols

#--- order
_dup
