# REQUIRES: sparc
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=sparc a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=sparc b.s -o b.o
# RUN: ld.lld -shared b.o -soname=b.so -o b.so
# RUN: ld.lld -shared a.o b.so -o a.so
# RUN: llvm-readelf -r a.so | FileCheck %s --check-prefix=GD-REL

## The 32-bit ABI uses the DTPMOD32/DTPOFF32/TPOFF32 GOT slots. a0 is hidden,
## so its offset is known at link time and only the module index needs a
## dynamic relocation.
# GD-REL:      Relocation section '.rela.dyn' {{.*}} contains 3 entries:
# GD-REL:      R_SPARC_TLS_DTPMOD32 0{{$}}
# GD-REL-DAG:  R_SPARC_TLS_DTPMOD32 {{.*}} b + 0
# GD-REL-DAG:  R_SPARC_TLS_DTPOFF32 {{.*}} b + 0
# GD-REL:      R_SPARC_JMP_SLOT {{.*}} __tls_get_addr + 0

## a0 is not preemptible in an executable, so its sequence becomes Local Exec.
## b stays preemptible and becomes Initial Exec, where the 32-bit ABI loads the
## offset with ld rather than V9's ldx.
# RUN: ld.lld a.o b.so -o a
# RUN: llvm-readelf -r a | FileCheck %s --check-prefix=EXE-REL
# RUN: llvm-objdump -d -j .text --no-show-raw-insn --no-print-imm-hex a | FileCheck %s --check-prefix=EXE

# EXE-REL:     R_SPARC_TLS_TPOFF32 {{.*}} b + 0
# EXE-REL-NOT: R_SPARC_TLS_DTPMOD32

## a0 is the only symbol in the TLS block, so a0 - tp = -4.
# EXE-LABEL:   <_start>:
# EXE-NEXT:      sethi 0, %o0
# EXE-NEXT:      xor %o0, -4, %o0
# EXE-NEXT:      add %g7, %o0, %o0
# EXE-NEXT:      nop
# EXE-NEXT:      nop
# EXE-NEXT:      sethi 0, %o1
# EXE-NEXT:      add %o1, 4, %o1
# EXE-NEXT:      ld [%l7+%o1], %o0
# EXE-NEXT:      add %g7, %o0, %o0
# EXE-NEXT:      nop

#--- a.s
.globl _start
_start:
  sethi %tgd_hi22(a0), %o0
  add   %o0, %tgd_lo10(a0), %o0
  add   %l7, %o0, %o0, %tgd_add(a0)
  call  __tls_get_addr, %tgd_call(a0)
   nop

  sethi %tgd_hi22(b), %o1
  add   %o1, %tgd_lo10(b), %o1
  add   %l7, %o1, %o0, %tgd_add(b)
  call  __tls_get_addr, %tgd_call(b)
   nop

.section .tbss,"awT",@nobits
.globl a0
.hidden a0
a0:
  .word 0

#--- b.s
.section .tbss,"awT",@nobits
.globl b
b:
  .word 0
