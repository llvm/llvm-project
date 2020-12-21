# REQUIRES: ppc
# RUN: split-file %s %t
# RUN: llvm-mc --triple=ppc64le %t/a.s --filetype=obj -o %t/a.o
# RUN: llvm-mc --triple=ppc64le %t/b.s --filetype=obj -o %t/b.o
# RUN: llvm-mc --triple=ppc64le %t/tga.s --filetype=obj -o %t/tga.o

## User code can call __tls_get_addr by specifying the tls_index parameter.
## We need to allow R_PPC64_REL24/R_PPC64_REL24_NOTOC referencing __tls_get_addr
## without a pairing R_PPC64_TLSGD/R_PPC64_TLSLD.
# RUN: ld.lld --shared --fatal-warnings %t/b.o -o /dev/null

## Warn missing R_PPC64_TLSGD/R_PPC64_TLSLD.
# RUN: ld.lld --shared %t/a.o -o %t.so 2>&1 | FileCheck %s --check-prefix=WARN
# RUN: llvm-objdump -d --no-leading-addr %t.so | FileCheck %s --check-prefix=DIS

# RUN: ld.lld %t/a.o %t/tga.o -o %t2 2>&1 | FileCheck %s --check-prefix=WARN
# RUN: llvm-readelf -x .got %t2 | FileCheck %s --check-prefix=HEX
# RUN: llvm-objdump -d --no-leading-addr %t2 | FileCheck %s --check-prefix=DIS

# WARN: warning: {{.*}}.o: disable TLS relaxation due to R_PPC64_GOT_TLS* relocations without R_PPC64_TLSGD/R_PPC64_TLSLD relocations

## .got+0: x is local - relaxed to LE - its DTPMOD/DTPREL slots are link-time constants.
## DTPMOD is 1. DTPREL is st_value-0x8000 = -0x8000.
## .got+16: DTPMOD/DTPREL for _TLS_MODULE_BASE_ is 1 and 0, respectively.
## .got+32: TPOFFSET for x = st_value-0x7000
# HEX:      section '.got':
# HEX-NEXT: [[#%x,IGNORE:]] 01000000 00000000 0080ffff ffffffff
# HEX-NEXT: [[#%x,IGNORE:]] 01000000 00000000 00000000 00000000
# HEX-NEXT: [[#%x,IGNORE:]] 0090ffff ffffffff

## .TOC.-32768 = (.got+0x8000)-32768 = .got
# DIS-LABEL: <GeneralDynamic>:
# DIS-NEXT:    addis 3, 2, 0
# DIS-NEXT:    addi 3, 3, -32768
# DIS-NEXT:    bl [[#%x,TGA:]]
# DIS-LABEL: <GeneralDynamic_NOTOC>:
# DIS-NEXT:    addis 3, 2, 0
# DIS-NEXT:    addi 3, 3, -32768
# DIS-NEXT:    bl [[#TGA]]

## LocalDynamic references _TLS_MODULE_BASE_.
## .TOC.-32752 = (.got+0x8000)-32752 = .got+16
# DIS-LABEL: <LocalDynamic>:
# DIS-NEXT:    addis 3, 2, 0
# DIS-NEXT:    addi 3, 3, -32752
# DIS-NEXT:    bl [[#TGA]]
# DIS-LABEL: <LocalDynamic_NOTOC>:
# DIS-NEXT:    addis 3, 2, 0
# DIS-NEXT:    addi 3, 3, -32752
# DIS-NEXT:    bl [[#TGA]]

## Technically we don't have to disable IE to LE relaxation,
## but disabling it for implementation simplicity does not hurt.
# DIS-LABEL: <InitialExec>:
# DIS-NEXT:    addis 3, 2, 0
# DIS-NEXT:    ld 3, -32736(3)
# DIS-NEXT:    add 3, 3, 13

#--- a.s
GeneralDynamic:
  addis 3, 2, x@got@tlsgd@ha
  addi 3, 3, x@got@tlsgd@l
  bl __tls_get_addr
  nop

GeneralDynamic_NOTOC:
  addis 3, 2, x@got@tlsgd@ha
  addi 3, 3, x@got@tlsgd@l
  bl __tls_get_addr@notoc
  nop

LocalDynamic:
  addis 3, 2, x@got@tlsld@ha
  addi 3, 3, x@got@tlsld@l
  bl __tls_get_addr
  nop

LocalDynamic_NOTOC:
  addis 3, 2, x@got@tlsld@ha
  addi 3, 3, x@got@tlsld@l
  bl __tls_get_addr@notoc
  nop

InitialExec:
  addis 3, 2, x@got@tprel@ha
  ld 3, x@got@tprel@l(3)
  add 3, 3, x@tls

.globl _start
_start:

.section .tbss,"awT",@nobits
.globl x
x:
  .quad 0

#--- b.s
CallOnly:
  bl __tls_get_addr
  nop
  blr

#--- tga.s
.globl __tls_get_addr
__tls_get_addr:
  blr
