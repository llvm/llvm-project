// This test checks if lld puts magic string "PGU" when an object files contains
// .pgu section.

// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-windows-msvc %s -o %t.main_x86.obj
// RUN: llvm-mc -filetype=obj -triple=aarch64-pc-windows-msvc %s -o %t.main_aarch.obj

// RUN: lld-link -out:%t_x86.exe %t.main_x86.obj  -entry:entry -subsystem:console -debug:symtab
// RUN: lld-link -out:%t_aarch.exe %t.main_aarch.obj  -entry:entry -subsystem:console -debug:symtab
// RUN: llvm-readobj --coff-debug-directory %t_x86.exe | FileCheck --check-prefix=CHECK_PGU %s
// RUN: llvm-readobj --coff-debug-directory %t_aarch.exe | FileCheck --check-prefix=CHECK_PGU %s
// CHECK_PGU: {{.*}}UGP{{.*}}

#--- main.s
.section .pgu
.global entry
entry:
