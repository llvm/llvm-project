# REQUIRES: x86
# RUN: llvm-mc -triple=x86_64-windows-gnu -filetype=obj -o %t.obj %s
# RUN: lld-link %t.obj /out:%t.exe /entry:main /subsystem:console /opt:ref
# RUN: llvm-readobj --file-headers %t.exe | FileCheck %s

# CHECK: TLSTableRVA: 0x1000
# CHECK: TLSTableSize: 0x28

  .section .text@main,"xr",one_only,main
  .globl main
main:
  ret

  .section .tls$aaa
tlsvar:
  .long 1

  .section .rdata$_tls_used,"dr",one_only,_tls_used
  .globl _tls_used
_tls_used:
  .zero 40
