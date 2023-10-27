# REQUIRES: x86

# RUN: llvm-mc -triple=x86_64-windows %s -filetype=obj -o %t.obj

# RUN: lld-link -dll -out:%t.dll -entry:entry %t.obj -subsystem:console 2>&1 | FileCheck --allow-empty --ignore-case %s

# CHECK-NOT: ignoring unknown argument
# CHECK-NOT: inferasanlibs
# CHECK-NOT: is not allowed in .drectve

  .global entry
  .text
entry:
  ret
  .section .drectve
  .ascii " /INFERASANLIBS "
