# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/movq.s -o %t/movq.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/leaq.s -o %t/leaq.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/notlv.s -o %t/notlv.o

# RUN: %lld -dylib -undefined dynamic_lookup -o %t/movq.dylib %t/movq.o
# RUN: llvm-objdump --macho --section-headers --bind -d --no-show-raw-insn %t/movq.dylib | \
# RUN:   FileCheck %s --check-prefix=MOVQ
# MOVQ-LABEL: _f:
# MOVQ-NEXT:  movq
# MOVQ-NOT:   __thread_ptrs
# MOVQ:       __got
# MOVQ-LABEL: Bind table:
# MOVQ:       __DATA_CONST __got 0x{{[0-9a-f]+}} pointer 0 flat-namespace _tlv

## FIXME: leaq computes the address of the slot itself, but the ABI requires the
## descriptor the slot holds, so this is wrong code. It is accepted because
## X86_64::relaxGotLoad's MOVQ check only runs on the relax path, and a slot
## exists here. Pre-existing and not specific to dynamic lookup: the same leaq
## against a dylib that really exports _tlv as thread-local miscompiles
## identically on stock lld. Pinned as-is so the separate fix has a baseline.
# RUN: %lld -dylib -undefined dynamic_lookup -o %t/leaq.dylib %t/leaq.o
# RUN: llvm-objdump --macho -d --no-show-raw-insn %t/leaq.dylib | \
# RUN:   FileCheck %s --check-prefix=LEAQ
# LEAQ-LABEL: _f:
# LEAQ-NEXT:  leaq

# RUN: %lld -dylib -install_name @rpath/libnotlv.dylib -o %t/libnotlv.dylib %t/notlv.o
# RUN: not %lld -dylib -o /dev/null %t/movq.o %t/libnotlv.dylib 2>&1 | \
# RUN:   FileCheck %s --check-prefix=DEFINED
# DEFINED: error: {{.*}}TLV relocation requires that symbol _tlv be thread-local

#--- movq.s
.globl _f
_f:
  movq _tlv@TLVP(%rip), %rax
  retq
.subsections_via_symbols

#--- leaq.s
.globl _f
_f:
  leaq _tlv@TLVP(%rip), %rax
  retq
.subsections_via_symbols

#--- notlv.s
.globl _tlv
.data
_tlv:
  .quad 0
