# REQUIRES: sparc
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=sparcv9 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=sparcv9 b.s -o b.o
# RUN: ld.lld -shared b.o -o b.so
# RUN: ld.lld -shared a.o b.so -o a.so
# RUN: llvm-readelf -r a.so | FileCheck %s --check-prefix=GD-REL
# RUN: llvm-objdump -d -j .text --no-show-raw-insn --no-print-imm-hex a.so | FileCheck %s --check-prefix=GD

## Each sequence gets a two-slot GOT entry holding the module index and the
## offset within that module's TLS block. a0 is hidden, so its offset is known
## at link time and only the module index needs a dynamic relocation. The call
## goes through the PLT even though the relocation names the TLS symbol rather
## than __tls_get_addr.
# GD-REL:      Relocation section '.rela.dyn' {{.*}} contains 3 entries:
# GD-REL:      R_SPARC_TLS_DTPMOD64 0{{$}}
# GD-REL-DAG:  R_SPARC_TLS_DTPMOD64 {{.*}} b + 0
# GD-REL-DAG:  R_SPARC_TLS_DTPOFF64 {{.*}} b + 0
# GD-REL:      Relocation section '.rela.plt' {{.*}} contains 1 entries:
# GD-REL:      R_SPARC_JMP_SLOT {{.*}} __tls_get_addr + 0

# GD-LABEL:   <_start>:
# GD-NEXT:      sethi 0, %o0
# GD-NEXT:      add %o0, 8, %o0
# GD-NEXT:      add %l7, %o0, %o0
# GD-NEXT:      call
# GD-NEXT:      nop
# GD-NEXT:      sethi 0, %o1
# GD-NEXT:      add %o1, 24, %o1
# GD-NEXT:      add %l7, %o1, %o0
# GD-NEXT:      call
# GD-NEXT:      nop

## In an executable a0 is not preemptible, so its sequence becomes Local Exec:
## the sethi holds the complement of the offset, the add becomes an xor, the
## GOT pointer becomes the thread pointer and the call is dropped. b is
## preemptible, so its sequence becomes Initial Exec instead: the add becomes
## the GOT load and the call becomes the thread-pointer add.
# RUN: ld.lld a.o b.so -o a
# RUN: llvm-readelf -r a | FileCheck %s --check-prefix=EXE-REL
# RUN: llvm-objdump -d -j .text --no-show-raw-insn --no-print-imm-hex a | FileCheck %s --check-prefix=EXE

# EXE-REL:     Relocation section '.rela.dyn' {{.*}} contains 1 entries:
# EXE-REL:     R_SPARC_TLS_TPOFF64 {{.*}} b + 0
# EXE-REL-NOT: R_SPARC_TLS_DTPMOD64
# EXE-REL-NOT: __tls_get_addr

# EXE-LABEL:   <_start>:
# EXE-NEXT:      sethi 0, %o0
# EXE-NEXT:      xor %o0, -8, %o0
# EXE-NEXT:      add %g7, %o0, %o0
# EXE-NEXT:      nop
# EXE-NEXT:      nop
# EXE-NEXT:      sethi 0, %o1
# EXE-NEXT:      add %o1, 8, %o1
# EXE-NEXT:      ldx [%l7+%o1], %o0
# EXE-NEXT:      add %g7, %o0, %o0
# EXE-NEXT:      nop

#--- a.s
.globl _start
_start:
## a0 is hidden, so it is not preemptible in an executable.
  sethi %tgd_hi22(a0), %o0
  add   %o0, %tgd_lo10(a0), %o0
  add   %l7, %o0, %o0, %tgd_add(a0)
  call  __tls_get_addr, %tgd_call(a0)
   nop

## b is defined in a DSO, so it stays preemptible in an executable.
  sethi %tgd_hi22(b), %o1
  add   %o1, %tgd_lo10(b), %o1
  add   %l7, %o1, %o0, %tgd_add(b)
  call  __tls_get_addr, %tgd_call(b)
   nop

.section .tbss,"awT",@nobits
.globl a0
.hidden a0
a0:
  .xword 0

#--- b.s
.section .tbss,"awT",@nobits
.globl b
b:
  .xword 0
