# REQUIRES: aarch64

# Tests that -dead_strip keeps only live ObjC class stubs and does not keep dead
# class definitions or dylib loads alive.

# RUN: rm -rf %t && split-file %s %t

# Check that only the live class-stub dependency survives dead stripping.
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/main.s -o %t/main.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/dylib.s \
# RUN:   -o %t/dylib.o
# RUN: %lld -arch arm64 -dylib -install_name @executable_path/libdead.dylib \
# RUN:   -o %t/libdead.dylib %t/dylib.o
# RUN: %lld -arch arm64 -lSystem -dead_strip -dead_strip_dylibs \
# RUN:   -o %t.out %t/main.o %t/libdead.dylib -U _objc_msgSend \
# RUN:   -objc_stubs_small
# RUN: llvm-objdump --no-show-raw-insn --section=__TEXT,__objc_stubs \
# RUN:   --macho %t.out | FileCheck %s --check-prefix=STUBS
# RUN: llvm-nm %t.out | FileCheck %s --check-prefix=SYMS
# RUN: llvm-otool -L %t.out | FileCheck %s --check-prefix=LOAD

# STUBS:      Contents of (__TEXT,__objc_stubs) section
# STUBS-NEXT: _objc_msgSendClass$live$_OBJC_CLASS_$_LiveClass:
# STUBS-NOT:  _objc_msgSendClass$dead$_OBJC_CLASS_$_DeadClass:
# STUBS-NOT:  _objc_msgSendClass$missing$_OBJC_CLASS_$_NeverDefined:
# STUBS-NOT:  _objc_msgSendClass$dylib$_OBJC_CLASS_$_DylibClass:

# SYMS-NOT: _OBJC_CLASS_$_DeadClass
# SYMS-NOT: _OBJC_CLASS_$_NeverDefined
# SYMS:     _OBJC_CLASS_$_LiveClass
# SYMS-NOT: _OBJC_CLASS_$_DeadClass
# SYMS-NOT: _OBJC_CLASS_$_NeverDefined

# LOAD-NOT: libdead.dylib
# LOAD:     /usr/lib/libSystem.dylib
# LOAD-NOT: libdead.dylib

#--- main.s
# Contains one live class stub and several dead class-stub references.
.text
.globl _main
_main:
  bl _live
  ret

.globl _live
_live:
  bl _objc_msgSendClass$live$_OBJC_CLASS_$_LiveClass
  ret

.globl _dead
_dead:
  bl _objc_msgSendClass$dead$_OBJC_CLASS_$_DeadClass
  ret

.globl _missing
_missing:
  bl _objc_msgSendClass$missing$_OBJC_CLASS_$_NeverDefined
  ret

.globl _dylibdead
_dylibdead:
  bl _objc_msgSendClass$dylib$_OBJC_CLASS_$_DylibClass
  ret

.data
.globl _OBJC_CLASS_$_LiveClass
_OBJC_CLASS_$_LiveClass:
  .quad 0

.globl _OBJC_CLASS_$_DeadClass
_OBJC_CLASS_$_DeadClass:
  .quad 0

.subsections_via_symbols

#--- dylib.s
# Provides a dylib class referenced only by dead code.
.data
.globl _OBJC_CLASS_$_DylibClass
_OBJC_CLASS_$_DylibClass:
  .quad 0
