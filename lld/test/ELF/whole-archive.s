// REQUIRES: x86

// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux \
// RUN:   %p/Inputs/whole-archive.s -o %ta.o
// RUN: rm -f %t.a
// RUN: llvm-ar rcs %t.a %ta.o

// Should not add symbols from the archive by default as they are not required
// RUN: ld.lld2 -o %t3 %t.o %t.a
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=NOTADDED %s
// NOTADDED: Symbols [
// NOTADDED-NOT: Name: _bar
// NOTADDED: ]

// Should add symbols from the archive if --whole-archive is used
// RUN: ld.lld2 -o %t3 %t.o --whole-archive %t.a
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=ADDED %s
// ADDED: Symbols [
// ADDED: Name: _bar
// ADDED: ]

// --no-whole-archive should restore default behaviour
// RUN: ld.lld2 -o %t3 %t.o --whole-archive --no-whole-archive %t.a
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=NOTADDED %s

// --whole-archive and --no-whole-archive should affect only archives which follow them
// RUN: ld.lld2 -o %t3 %t.o %t.a --whole-archive --no-whole-archive
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=NOTADDED %s
// RUN: ld.lld2 -o %t3 %t.o --whole-archive %t.a --no-whole-archive
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=ADDED %s

.globl _start;
_start:
