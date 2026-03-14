# REQUIRES: x86

# RUN: llvm-mc -triple=x86_64-windows %s -filetype=obj -o %t.obj

# RUN: not lld-link -dll -out:%t.dll -entry:entry %t.obj -subsystem:console 2>&1 | FileCheck %s

# CHECK: warning: ignoring unknown argument: -unknowndirectivename
# CHECK: error: -unknowndirectivename is not allowed in .drectve ({{.*}}.obj)

  .global entry
  .text
entry:
  ret
  .section .drectve
  .ascii " -unknowndirectivename "
