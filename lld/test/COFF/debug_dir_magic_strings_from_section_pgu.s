// This test checks if lld puts magic string "PGU" reversed when an object files contains
// .pgu section.

// REQUIRES: system-windows

// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-windows-msvc %s -o %t.main.obj

// RUN: lld-link -out:%t.exe %t.main.obj  -entry:entry -subsystem:console -debug:symtab
// RUN: dumpbin /HEADERS %t.exe
// CHECK: PGU

#--- main.s
.section .pgu
.global entry
entry:
  movl %edx, %edx

