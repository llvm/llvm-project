# REQUIRES: aarch64
# RUN: yaml2obj %s -o %t.o
# RUN: not ld.lld --cmse-implib %t.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR_CMSE_IMPLIB
# RUN: not ld.lld --in-implib=%t.o %t.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR_IN_IMPLIB
# RUN: not ld.lld --out-implib=out.lib %t.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR_OUT_IMPLIB

# ERR_CMSE_IMPLIB: error: --cmse-implib is only supported on ARM targets
# ERR_IN_IMPLIB: error: --in-implib is only supported on ARM targets
# ERR_OUT_IMPLIB: error: --out-implib is only supported on ARM targets

!ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_REL
  Machine:         EM_AARCH64
