// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=i686-win32-gnu %s -o %t.o

// RUN: lld-link -lldmingw -dll -out:%t.dll %t.o -noentry
// RUN: llvm-readobj --coff-exports %t.dll | FileCheck --implicit-check-not=Name: %s

// CHECK: Name: sym1

.global _sym1
_sym1:
  ret

.global _sym2
_sym2:
  ret

.global _sym3
_sym3:
  ret

.section .drectve,"yn"
.ascii " -exclude-symbols:sym2,unknownsym"
.ascii " -exclude-symbols:unkonwnsym,sym3"
