# REQUIRES: ppc
# RUN: llvm-mc --triple=powerpc64le %s --filetype=obj -o %t1.o
# RUN: llvm-mc --triple=powerpc64 %s --filetype=obj -o %t2.o
# RUN: ld.lld --shared --fatal-warnings %t1.o -o /dev/null
# RUN: ld.lld --shared --fatal-warnings %t2.o -o /dev/null

## User code can call __tls_get_addr by specifying the tls_index parameter.
## We need to allow R_PPC64_REL24/R_PPC64_REL24_NOTOC referencing __tls_get_addr
## without a pairing R_PPC64_TLSGD/R_PPC64_TLSLD.

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
