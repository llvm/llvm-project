# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/weak-foo.s -o %t/weak-foo.o

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/weak-autohide-foo.s -o %t/weak-autohide-foo.o

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/weak-foo-pe.s -o %t/weak-foo-pe.o

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/weak-autohide-foo-pe.s -o %t/weak-autohide-foo-pe.o

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/ref-foo.s -o %t/ref-foo.o

## Basics: A weak_def_can_be_hidden symbol should not be in the output's
## export table, nor in the weak bind table, and references to it from
## within the dylib should not use weak indirect lookups.
## Think: Inline function compiled without any -fvisibility flags, inline
## function does not have its address taken. In -O2 compiles, GlobalOpt will
## upgrade the inline function from .weak_definition to .weak_def_can_be_hidden.
# RUN: %lld -dylib -o %t/weak-autohide-foo.dylib \
# RUN:     %t/weak-autohide-foo.o %t/ref-foo.o
# RUN: llvm-objdump --syms --exports-trie %t/weak-autohide-foo.dylib | \
# RUN:     FileCheck --check-prefix=EXPORTS %s
# RUN: llvm-nm -m %t/weak-autohide-foo.dylib | \
# RUN:     FileCheck --check-prefix=EXPORTS-NM %s
# RUN: llvm-objdump --macho --bind --weak-bind %t/weak-autohide-foo.dylib | \
# RUN:     FileCheck --check-prefix=WEAKBIND %s
# RUN: llvm-objdump --macho --private-header %t/weak-autohide-foo.dylib | \
# RUN:     FileCheck --check-prefix=HEADERS %s

## ld64 doesn't treat .weak_def_can_be_hidden as weak symbols in the
## exports trie or bind tables, but it claims they are weak in the symbol
## table. lld marks them as local in the symbol table, see also test
## weak-private-extern.s. These FileCheck lines match both outputs.
# EXPORTS-LABEL: SYMBOL TABLE:
# EXPORTS-DAG:   [[#%x, FOO_ADDR:]] {{.*}} _foo
# EXPORTS-LABEL: Exports trie:
# EXPORTS-NOT:   0x{{0*}}[[#%X, FOO_ADDR]] _foo

## nm output for .weak_def_can_be_hidden says "was a private external" even
## though it wasn't .private_extern: It was just .weak_def_can_be_hidden.
## This matches ld64.
# EXPORTS-NM: (__TEXT,__text) non-external (was a private external) _foo

# WEAKBIND-NOT: __got
# WEAKBIND-NOT: __la_symbol_ptr

## ld64 sets WEAKBIND and BINDS_TO_WEAK in the mach-o header even though there
## are no weak bindings or weak definitions after processing the autohide. That
## looks like a bug in ld64 (?) If you change lit.local.cfg to set %lld to ld to
## test compatibility, you have to add some arbitrary suffix to these two lines:
# HEADERS-NOT: WEAK_DEFINES
# HEADERS-NOT: BINDS_TO_WEAK

## Same behavior for a symbol that's both .weak_def_can_be_hidden and
## .private_extern. Think: Inline function compiled with
## -fvisibility-inlines-hidden.
# RUN: %lld -dylib -o %t/weak-autohide-foo-pe.dylib \
# RUN:     %t/weak-autohide-foo-pe.o %t/ref-foo.o
# RUN: llvm-objdump --syms --exports-trie %t/weak-autohide-foo-pe.dylib | \
# RUN:     FileCheck --check-prefix=EXPORTS %s
# RUN: llvm-nm -m %t/weak-autohide-foo-pe.dylib | \
# RUN:     FileCheck --check-prefix=EXPORTS-NM %s
# RUN: llvm-objdump --macho --bind --weak-bind %t/weak-autohide-foo-pe.dylib | \
# RUN:     FileCheck --check-prefix=WEAKBIND %s
# RUN: llvm-objdump --macho --private-header %t/weak-autohide-foo-pe.dylib | \
# RUN:     FileCheck --check-prefix=HEADERS %s

## In fact, a regular weak symbol that's .private_extern behaves the same
## as well.
# RUN: %lld -dylib -o %t/weak-foo-pe.dylib %t/weak-foo-pe.o %t/ref-foo.o
# RUN: llvm-objdump --syms --exports-trie %t/weak-foo-pe.dylib | \
# RUN:     FileCheck --check-prefix=EXPORTS %s
# RUN: llvm-nm -m %t/weak-foo-pe.dylib | \
# RUN:     FileCheck --check-prefix=EXPORTS-NM %s
# RUN: llvm-objdump --macho --bind --weak-bind %t/weak-foo-pe.dylib | \
# RUN:     FileCheck --check-prefix=WEAKBIND %s
# RUN: llvm-objdump --macho --private-header %t/weak-foo-pe.dylib | \
# RUN:     FileCheck --check-prefix=HEADERS %s

## Combining a regular weak_definition with a weak_def_can_be_hidden produces
## a regular weak external.
# RUN: %lld -dylib -o %t/weak-foo.dylib -lSystem \
# RUN:     %t/weak-autohide-foo.o %t/weak-foo.o %t/ref-foo.o
# RUN: llvm-objdump --syms --exports-trie %t/weak-foo.dylib | \
# RUN:     FileCheck --check-prefix=WEAK %s
# RUN: llvm-nm -m %t/weak-foo.dylib | \
# RUN:     FileCheck --check-prefix=WEAK-NM %s
# RUN: llvm-objdump --macho --bind --weak-bind %t/weak-foo.dylib | \
# RUN:     FileCheck --check-prefix=WEAK-WEAKBIND %s
# RUN: llvm-objdump --macho --private-header %t/weak-foo.dylib | \
# RUN:     FileCheck --check-prefix=WEAK-HEADERS %s
# WEAK-LABEL: SYMBOL TABLE:
# WEAK-DAG:   [[#%x, FOO_ADDR:]] w {{.*}} _foo
# WEAK-LABEL: Exports trie:
# WEAK-DAG:   0x{{0*}}[[#%X, FOO_ADDR]] _foo
# WEAK-NM: (__TEXT,__text) weak external _foo
# WEAK-WEAKBIND: __la_symbol_ptr 0x{{.*}} pointer 0           _foo
# WEAK-HEADERS: WEAK_DEFINES
# WEAK-HEADERS: BINDS_TO_WEAK

#--- weak-foo.s
.globl _foo
.weak_definition _foo
_foo:
  retq

#--- weak-autohide-foo.s
.globl _foo
.weak_def_can_be_hidden _foo
_foo:
  retq

# An alias is the only way to set .weak_def_can_be_hidden on an already-hidden symbol.
# Veryify that LLD can handle these double-hidden symbols gracefully.
.set l_foo, _foo

#--- weak-foo-pe.s
.private_extern _foo
.globl _foo
.weak_definition _foo
_foo:
  retq

#--- weak-autohide-foo-pe.s
.private_extern _foo
.globl _foo
.weak_def_can_be_hidden _foo
_foo:
  retq

#--- ref-foo.s
.globl _bar
_bar:
  callq _foo
  retq
