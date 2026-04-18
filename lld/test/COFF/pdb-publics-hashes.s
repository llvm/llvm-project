# REQUIRES: x86
# RUN: llvm-mc -filetype=obj %s -o %t.obj -triple x86_64-windows-msvc
# RUN: lld-link -entry:wWinMain -nodefaultlib %t.obj -out:%t.exe -pdb:%t.pdb -debug
# RUN: llvm-pdbutil dump -publics -public-extras %t.pdb | FileCheck %s

# Check that each symbol is placed in its own hash bucket.
# CHECK: Public Symbols
# CHECK: Hash Buckets
# CHECK-NEXT: 0x00000000
# CHECK-NEXT: 0x0000000c
# CHECK-NEXT: 0x00000018

.globl foo
foo:
.rept 33
nop
.endr

# An 8-byte symbol name, meaning it fits precisely in the COFF symbol table's
# name field and will not be null terminated. It's followed by the symbol
# address in little-endian: 33 (ascii '!'), meaning if the symbol hashing
# assumes this name is null terminated, it will compute the same hash as for
# the symbol below, putting them in the same bucket.
.globl wWinMain
wWinMain:
nop

.globl "wWinMain!"
"wWinMain!":
nop
