# REQUIRES: ppc
# RUN: llvm-mc --triple=powerpc64le %s --filetype=obj -o %t1.o
# RUN: llvm-mc --triple=powerpc64 %s --filetype=obj -o %t2.o
# RUN: not ld.lld --shared %t1.o -o /dev/null 2>&1 | FileCheck %s
# RUN: not ld.lld --shared %t2.o -o /dev/null 2>&1 | FileCheck %s

# CHECK:      ld.lld: error: call to __tls_get_addr is missing a R_PPC64_TLSGD/R_PPC64_TLSLD relocation
# CHECK-NEXT:   defined in {{.*}}.o
# CHECK-NEXT:   referenced by {{.*}}.o:(.text+0x8)

# CHECK:      ld.lld: error: call to __tls_get_addr is missing a R_PPC64_TLSGD/R_PPC64_TLSLD relocation
# CHECK-NEXT:   defined in {{.*}}.o
# CHECK-NEXT:   referenced by {{.*}}.o:(.text+0x18)

# CHECK:      ld.lld: error: call to __tls_get_addr is missing a R_PPC64_TLSGD/R_PPC64_TLSLD relocation
# CHECK-NEXT:   defined in {{.*}}.o
# CHECK-NEXT:   referenced by {{.*}}.o:(.text+0x28)

# CHECK:      ld.lld: error: call to __tls_get_addr is missing a R_PPC64_TLSGD/R_PPC64_TLSLD relocation
# CHECK-NEXT:   defined in {{.*}}.o
# CHECK-NEXT:   referenced by {{.*}}.o:(.text+0x38)

# CHECK:      ld.lld: error: call to __tls_get_addr is missing a R_PPC64_TLSGD/R_PPC64_TLSLD relocation
# CHECK-NEXT:   defined in {{.*}}.o
# CHECK-NEXT:   referenced by {{.*}}.o:(.text+0x40)

GeneralDynamic:
  addis 3, 2, x@got@tlsgd@ha
  addi 3, 3, x@got@tlsgd@l
  bl __tls_get_addr
  blr

GeneralDynamic_NOTOC:
  addis 3, 2, x@got@tlsgd@ha
  addi 3, 3, x@got@tlsgd@l
  bl __tls_get_addr@notoc
  blr

LocalDynamic:
  addis 3, 2, x@got@tlsld@ha
  addi 3, 3, x@got@tlsld@l
  bl __tls_get_addr
  blr

LocalDynamic_NOTOC:
  addis 3, 2, x@got@tlsld@ha
  addi 3, 3, x@got@tlsld@l
  bl __tls_get_addr@notoc
  blr

CallOnly:
  bl __tls_get_addr
  blr
