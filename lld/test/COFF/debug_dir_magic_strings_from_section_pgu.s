// This test checks if lld puts magic string "PGU" when an object files contains
// .pgu section.

// REQUIRES: x86

// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-windows %s -o %t.main_x86.obj

// RUN: lld-link -out:%t_x86.exe %t.main_x86.obj  -entry:entry -subsystem:console -debug:symtab
// RUN: llvm-readobj --coff-debug-directory %t_x86.exe | FileCheck --check-prefix=CHECK_PGU %s
// CHECK_PGU: {{.*}}UGP{{.*}}

#--- main.s
.section .pgu
.global entry
entry:
