# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 %s -o %t.o
# RUN: %lld -pie -lSystem %t.o -o %t
# RUN: llvm-objdump --macho --unwind-info --rebase %t | FileCheck %s

## Check that we do not add rebase opcodes to the compact unwind section.
# CHECK:      Contents of __unwind_info section:
# CHECK-NEXT:   Version:                                   0x1
# CHECK-NEXT:   Common encodings array section offset:
# CHECK-NEXT:   Number of common encodings in array:       0x1
# CHECK:      Rebase table:
# CHECK-NEXT: segment  section            address     type
# CHECK-EMPTY:

.globl _main
.text
_main:
  .cfi_startproc
  .cfi_def_cfa_offset 16
  retq
  .cfi_endproc
