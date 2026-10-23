# REQUIRES: sparc
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=sparcv9 a.s -o a.o
# RUN: ld.lld -shared a.o -o a.so
# RUN: llvm-readelf -r a.so | FileCheck %s --check-prefix=LD-REL
# RUN: llvm-objdump -d -j .text --no-show-raw-insn --no-print-imm-hex a.so | FileCheck %s --check-prefix=LD

## Local Dynamic needs only the module index, so one GOT entry and one dynamic
## relocation serve every symbol in the module. The per-symbol offsets are
## link-time constants.
# LD-REL:      Relocation section '.rela.dyn' {{.*}} contains 1 entries:
# LD-REL:      R_SPARC_TLS_DTPMOD64 0{{$}}
# LD-REL:      Relocation section '.rela.plt' {{.*}} contains 1 entries:
# LD-REL:      R_SPARC_JMP_SLOT {{.*}} __tls_get_addr + 0

# LD-LABEL:   <_start>:
# LD-NEXT:      sethi 0, %o0
# LD-NEXT:      add %o0, 8, %o0
# LD-NEXT:      add %l7, %o0, %o0
# LD-NEXT:      call
# LD-NEXT:      nop
## a0 is at offset 0 in the TLS block, a1 at 8.
# LD-NEXT:      sethi 0, %o1
# LD-NEXT:      xor %o1, 0, %o1
# LD-NEXT:      add %o0, %o1, %o2
# LD-NEXT:      sethi 0, %o3
# LD-NEXT:      xor %o3, 8, %o3
# LD-NEXT:      add %o0, %o3, %o4

## In an executable the whole sequence becomes Local Exec: the module-index
## setup is dropped, the call loads the thread pointer into %o0, and the
## per-symbol offsets become the negative @tpoff, encoded as a complement.
# RUN: ld.lld a.o -o a
# RUN: llvm-readelf -r a | FileCheck %s --check-prefix=LE-REL
# RUN: llvm-objdump -d -j .text --no-show-raw-insn --no-print-imm-hex a | FileCheck %s --check-prefix=LE

# LE-REL-NOT:  R_SPARC_TLS
# LE-REL-NOT:  __tls_get_addr

# LE-LABEL:   <_start>:
# LE-NEXT:      nop
# LE-NEXT:      nop
# LE-NEXT:      nop
# LE-NEXT:      mov %g7, %o0
# LE-NEXT:      nop
## a0 - tp = -0x10, a1 - tp = -8.
# LE-NEXT:      sethi 0, %o1
# LE-NEXT:      xor %o1, -16, %o1
# LE-NEXT:      add %o0, %o1, %o2
# LE-NEXT:      sethi 0, %o3
# LE-NEXT:      xor %o3, -8, %o3
# LE-NEXT:      add %o0, %o3, %o4

#--- a.s
.globl _start
_start:
  sethi %tldm_hi22(a0), %o0
  add   %o0, %tldm_lo10(a0), %o0
  add   %l7, %o0, %o0, %tldm_add(a0)
  call  __tls_get_addr, %tldm_call(a0)
   nop

  sethi %tldo_hix22(a0), %o1
  xor   %o1, %tldo_lox10(a0), %o1
  add   %o0, %o1, %o2, %tldo_add(a0)

  sethi %tldo_hix22(a1), %o3
  xor   %o3, %tldo_lox10(a1), %o3
  add   %o0, %o3, %o4, %tldo_add(a1)

.section .tbss,"awT",@nobits
.globl a0, a1
.hidden a0
.hidden a1
a0:
  .xword 0
a1:
  .xword 0
