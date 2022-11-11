// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t

// Should use "_init" and "_fini" by default when fills dynamic table
// RUN: ld.lld -shared %t -o %t2
// RUN: llvm-readobj --dynamic-table %t2 | FileCheck --check-prefix=BYDEF %s
// BYDEF: INIT 0x11010
// BYDEF: FINI 0x11020

// -init and -fini override symbols to use
// RUN: ld.lld -shared %t -o %t2 -init _foo -fini _bar
// RUN: llvm-readobj --dynamic-table %t2 | FileCheck --check-prefix=OVR %s
// OVR: INIT 0x11030
// OVR: FINI 0x11040

// Check aliases as well
// RUN: ld.lld -shared %t -o %t2 -init=_foo -fini=_bar
// RUN: llvm-readobj --dynamic-table %t2 | FileCheck --check-prefix=OVR %s

// Don't add an entry for undef. The freebsd dynamic linker doesn't
// check if the value is null. If it is, it will just call the
// load address.
// RUN: ld.lld -shared %t -o %t2 -init=_undef -fini=_undef
// RUN: llvm-readobj --dynamic-table %t2 | FileCheck --check-prefix=UNDEF %s
// UNDEF-NOT: INIT
// UNDEF-NOT: FINI

// Don't add an entry for shared. For the same reason as undef.
// RUN: ld.lld -shared %t -o %t.so
// RUN: echo > %t.s
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %t.s -o %t2.o
// RUN: ld.lld -shared %t2.o %t.so -o %t2
// RUN: llvm-readobj --dynamic-table %t2 | FileCheck --check-prefix=SHARED %s
// SHARED-NOT: INIT
// SHARED-NOT: FINI

// Should not add new entries to the symbol table
// and should not require given symbols to be resolved
// RUN: ld.lld -shared %t -o %t2 -init=_unknown -fini=_unknown
// RUN: llvm-readobj --symbols --dynamic-table %t2 | FileCheck --check-prefix=NOENTRY %s
// NOENTRY: DynamicSection [
// NOENTRY-NOT: INIT
// NOENTRY-NOT: FINI
// NOENTRY: ]
// NOENTRY: Symbols [
// NOENTRY-NOT: Name: _unknown
// NOENTRY: ]

// Should not add entries for "_init" and "_fini" to the symbol table
// if the symbols are defined in non-fetched archive members.
// RUN: rm -f %t.a
// RUN: llvm-ar rcs %t.a %t
// RUN: ld.lld -shared -m elf_x86_64 -e _unknown %t.a -o %t.so
// RUN: llvm-nm %t.so | \
// RUN:   FileCheck %s --implicit-check-not=_init --implicit-check-not=_fini

.global _start,_init,_fini,_foo,_bar,_undef
_start:
_init = 0x11010
_fini = 0x11020
_foo  = 0x11030
_bar  = 0x11040
