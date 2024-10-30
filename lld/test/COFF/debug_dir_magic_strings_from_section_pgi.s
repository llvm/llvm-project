// This test checks if lld puts magic string "PGI" when an object files contains
// .pgi section.

// REQUIRES: system-windows

// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-windows-msvc %s -o %t.main.obj

// RUN: lld-link -out:%t.exe %t.main.obj  -entry:entry -subsystem:console -debug:symtab
// RUN: dumpbin /HEADERS %t.exe
// CHECK: PGI

#--- main.s
.section .pgi
.global entry
entry:
  movl %edx, %edx

