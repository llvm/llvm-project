# REQUIRES: x86
## --wrap may trigger archive extraction. Test that local symbols are initialized.

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/a.s -o %t/a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/b.s -o %t/b.o
# RUN: ld.lld %t/a.o --start-lib %t/b.o --end-lib -o %t/a --wrap pthread_create -o /dev/null

#--- a.s
.globl _start
_start:
.cfi_startproc
  call pthread_create
.cfi_endproc

#--- b.s
.global __wrap_pthread_create
__wrap_pthread_create:
.cfi_startproc
  ret
.cfi_endproc
