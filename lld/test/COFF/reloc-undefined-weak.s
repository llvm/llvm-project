// REQUIRES: x86

// Check that base-relocations for unresolved weak symbols will be omitted.

// RUN: rm -rf %t.dir && split-file %s %t.dir && cd %t.dir
// RUN: llvm-mc -filetype=obj -triple=x86_64-mingw main.s -o main.obj
// RUN: llvm-mc -filetype=obj -triple=x86_64-mingw other.s -o other.obj

// RUN: lld-link -lldmingw -machine:x64 -dll -out:other.dll other.obj -noentry -export:foo -implib:other.lib
// RUN: lld-link -lldmingw -machine:x64 -out:main.exe main.obj other.lib -entry:entry -wrap:foo -debug:symtab
// RUN: llvm-readobj --sections --symbols --coff-imports --coff-basereloc main.exe | FileCheck %s --implicit-check-not=other.dll

// CHECK:      Number: 3
// CHECK-NEXT: Name: .data
// CHECK-NEXT: VirtualSize:
// CHECK-NEXT: VirtualAddress: 0x[[#%x,SECTOP:0x3000]]
// CHECK:      Name: ref_foo
// CHECK-NEXT: Value: [[#%d,SYMVAL:]]
// CHECK:      BaseReloc [
// CHECK-NOT:    Address: 0x[[#%x,SECTOP+SYMVAL]]

#--- main.s
.global entry
entry:
  movq ref_foo(%rip), %rax
  call *%rax

.global __wrap_foo
__wrap_foo:
  ret

.data
.global ref_foo
.p2align 3
ref_foo:
  .quad __real_foo

.globl _pei386_runtime_relocator
_pei386_runtime_relocator:
  movl __RUNTIME_PSEUDO_RELOC_LIST__(%rip), %eax
  movl __RUNTIME_PSEUDO_RELOC_LIST_END__(%rip), %eax

.weak __real_foo
.addrsig
.addrsig_sym __real_foo
.addrsig_sym ref_foo

#--- other.s
.global foo

foo:
  ret
