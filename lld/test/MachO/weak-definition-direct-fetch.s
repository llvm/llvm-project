# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t

## This test exercises the various possible combinations of weak and non-weak
## symbols that get referenced directly by a relocation in an object file.

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/refs-foo.s -o %t/refs-foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/weak-refs-foo.s -o %t/weak-refs-foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/weakfoo.s -o %t/weakfoo.o

# RUN: %lld -dylib -install_name \
# RUN:   @executable_path/libfoo.dylib %t/foo.o -o %t/libfoo.dylib
# RUN: %lld -dylib -install_name \
# RUN:   @executable_path/libweakfoo.dylib %t/weakfoo.o -o %t/libweakfoo.dylib

# RUN: llvm-objdump --macho --exports-trie %t/libweakfoo.dylib | FileCheck %s --check-prefix WEAK-DYLIB-CHECK
# WEAK-DYLIB-CHECK: _foo [weak_def]

## Make sure we are using the export trie and not the symbol table when linking
## against these dylibs.
# RUN: llvm-strip %t/libfoo.dylib
# RUN: llvm-strip %t/libweakfoo.dylib
# RUN: llvm-nm %t/libfoo.dylib 2>&1 | FileCheck %s --check-prefix=NOSYM
# RUN: llvm-nm %t/libweakfoo.dylib 2>&1 | FileCheck %s --check-prefix=NOSYM
# NOSYM: no symbols

# RUN: llvm-ar --format=darwin rcs %t/foo.a %t/foo.o
# RUN: llvm-ar --format=darwin rcs %t/weakfoo.a %t/weakfoo.o

## End of input file setup. The following lines check which symbol "wins" when
## there are multiple definitions.

# PREFER-NONWEAK-DYLIB:  __DATA __la_symbol_ptr 0x{{[0-9a-f]+}} libfoo _foo
# PREFER-WEAK-OBJECT:    O __TEXT,weak _foo
# PREFER-NONWEAK-OBJECT: O __TEXT,nonweak _foo
# NO-SYM-NOT: _foo

## First, we test the cases where the symbols are of the same type (both from a
## dylib, or both from an archive, etc.)
##
## For dylibs and object files, the non-weak symbol always wins. But the weak
## flag has no effect when we are dealing with two archive symbols.

# RUN: %lld -lSystem -o %t/weak-nonweak-dylibs -L%t -lweakfoo -lfoo %t/refs-foo.o
# RUN: llvm-objdump --macho --lazy-bind --syms %t/weak-nonweak-dylibs | FileCheck %s --check-prefix=PREFER-NONWEAK-DYLIB
# RUN: %lld -lSystem -o %t/nonweak-weak-dylibs -L%t -lfoo -lweakfoo %t/refs-foo.o
# RUN: llvm-objdump --macho --lazy-bind --syms %t/nonweak-weak-dylibs | FileCheck %s --check-prefix=PREFER-NONWEAK-DYLIB

# RUN: %lld -lSystem -o %t/weak-nonweak-objs %t/weakfoo.o %t/foo.o %t/refs-foo.o
# RUN: llvm-objdump --macho --lazy-bind --syms %t/weak-nonweak-objs | FileCheck %s --check-prefix=PREFER-NONWEAK-OBJECT
# RUN: %lld -lSystem -o %t/nonweak-weak-objs %t/foo.o %t/weakfoo.o %t/refs-foo.o
# RUN: llvm-objdump --macho --lazy-bind --syms %t/nonweak-weak-objs | FileCheck %s --check-prefix=PREFER-NONWEAK-OBJECT

# RUN: %lld -lSystem -o %t/weak-nonweak-archives %t/weakfoo.a %t/foo.a %t/refs-foo.o
# RUN: llvm-objdump --macho --lazy-bind --syms %t/weak-nonweak-archives | FileCheck %s --check-prefix=PREFER-WEAK-OBJECT
# RUN: %lld -lSystem -o %t/nonweak-weak-archives %t/foo.a %t/weakfoo.a %t/refs-foo.o
# RUN: llvm-objdump --macho --lazy-bind --syms %t/nonweak-weak-archives | FileCheck %s --check-prefix=PREFER-NONWEAK-OBJECT

## The next 5 chunks refs-foo.symbol pairs of different types.

## (Weak) archive symbols take precedence over weak dylib symbols.
# RUN: %lld -lSystem -o %t/weak-dylib-weak-ar -L%t -lweakfoo %t/weakfoo.a %t/refs-foo.o
# RUN: llvm-objdump --macho --lazy-bind --syms %t/weak-dylib-weak-ar | FileCheck %s --check-prefix=PREFER-WEAK-OBJECT
# RUN: %lld -lSystem -o %t/weak-ar-weak-dylib -L%t %t/weakfoo.a -lweakfoo %t/refs-foo.o
# RUN: llvm-objdump --macho --lazy-bind --syms %t/weak-ar-weak-dylib | FileCheck %s --check-prefix=PREFER-WEAK-OBJECT

## (Weak) archive symbols have the same precedence as dylib symbols.
# RUN: %lld -lSystem -o %t/weak-ar-nonweak-dylib -L%t %t/weakfoo.a -lfoo %t/refs-foo.o
# RUN: llvm-objdump --macho --lazy-bind --syms %t/weak-ar-nonweak-dylib | FileCheck %s --check-prefix=PREFER-WEAK-OBJECT
# RUN: %lld -lSystem -o %t/nonweak-dylib-weak-ar -L%t -lfoo %t/weakfoo.a %t/refs-foo.o
# RUN: llvm-objdump --macho --lazy-bind --syms %t/nonweak-dylib-weak-ar | FileCheck %s --check-prefix=PREFER-NONWEAK-DYLIB

## Weak defined symbols take precedence over weak dylib symbols.
# RUN: %lld -lSystem -o %t/weak-dylib-weak-obj -L%t -lweakfoo %t/weakfoo.o %t/refs-foo.o
# RUN: llvm-objdump --macho --lazy-bind --syms %t/weak-dylib-weak-obj | FileCheck %s --check-prefix=PREFER-WEAK-OBJECT
# RUN: %lld -lSystem -o %t/weak-obj-weak-dylib -L%t %t/weakfoo.o -lweakfoo %t/refs-foo.o
# RUN: llvm-objdump --macho --lazy-bind --syms %t/weak-obj-weak-dylib | FileCheck %s --check-prefix=PREFER-WEAK-OBJECT

## Weak defined symbols take precedence over dylib symbols.
# RUN: %lld -lSystem -o %t/weak-obj-nonweak-dylib -L%t %t/weakfoo.o -lfoo %t/refs-foo.o
# RUN: llvm-objdump --macho --lazy-bind --syms %t/weak-obj-nonweak-dylib | FileCheck %s --check-prefix=PREFER-WEAK-OBJECT
# RUN: %lld -lSystem -o %t/nonweak-dylib-weak-obj -L%t -lfoo %t/weakfoo.o %t/refs-foo.o
# RUN: llvm-objdump --macho --lazy-bind --syms %t/nonweak-dylib-weak-obj | FileCheck %s --check-prefix=PREFER-WEAK-OBJECT

## Weak defined symbols take precedence over archive symbols.
# RUN: %lld -lSystem -o %t/weak-obj-nonweak-ar %t/weakfoo.o %t/foo.a %t/refs-foo.o
# RUN: llvm-objdump --macho --lazy-bind --syms %t/weak-obj-nonweak-ar | FileCheck %s --check-prefix=PREFER-WEAK-OBJECT
# RUN: %lld -lSystem -o %t/nonweak-ar-weak-obj %t/foo.a %t/weakfoo.o %t/refs-foo.o
# RUN: llvm-objdump --macho --lazy-bind --syms %t/nonweak-ar-weak-obj | FileCheck %s --check-prefix=PREFER-WEAK-OBJECT

## Regression test: A weak dylib symbol that isn't referenced by an undefined
## symbol should not cause an archive symbol to get loaded.
# RUN: %lld -dylib -lSystem -o %t/weak-ar-weak-unref-dylib -L%t %t/weakfoo.a -lweakfoo
# RUN: llvm-objdump --macho --lazy-bind --syms %t/weak-ar-weak-unref-dylib | FileCheck %s --check-prefix=NO-SYM
# RUN: %lld -dylib -lSystem -o %t/weak-unref-dylib-weak-ar -L%t -lweakfoo %t/weakfoo.a
# RUN: llvm-objdump --macho --lazy-bind --syms %t/weak-unref-dylib-weak-ar | FileCheck %s --check-prefix=NO-SYM

## However, weak references are sufficient to cause the archive to be loaded.
# RUN: %lld -dylib -lSystem -o %t/weak-ar-weak-ref-weak-dylib -L%t %t/weakfoo.a -lweakfoo %t/weak-refs-foo.o
# RUN: llvm-objdump --macho --lazy-bind --syms %t/weak-ar-weak-ref-weak-dylib | FileCheck %s --check-prefix=PREFER-WEAK-OBJECT
# RUN: %lld -dylib -lSystem -o %t/weak-ref-weak-dylib-weak-ar -L%t -lweakfoo %t/weakfoo.a %t/weak-refs-foo.o
# RUN: llvm-objdump --macho --lazy-bind --syms %t/weak-ref-weak-dylib-weak-ar | FileCheck %s --check-prefix=PREFER-WEAK-OBJECT

#--- foo.s
.globl _foo
.section __TEXT,nonweak
_foo:

#--- weakfoo.s
.globl _foo
.weak_definition _foo
.section __TEXT,weak
_foo:

#--- refs-foo.s
.globl _main
_main:
  callq _foo
  ret

#--- weak-refs-foo.s
.globl _main
.weak_reference _foo
_main:
  callq _foo
  ret
