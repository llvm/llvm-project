# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/archive-foo.s -o %t/archive-foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/archive-baz.s -o %t/archive-baz.o
# RUN: llvm-ar rcs %t/libfoo.a %t/archive-foo.o
# RUN: llvm-ar rcs %t/libbaz.a %t/archive-baz.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/obj.s -o %t/obj.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/obj-bar.s -o %t/obj-bar.o

# RUN: %lld -dylib -lSystem %t/obj.o -load_hidden %t/libfoo.a -o %t/test.dylib
# RUN: llvm-nm %t/test.dylib | FileCheck %s
# RUN: %lld -dylib -lSystem -L%t %t/obj.o -hidden-lfoo -o %t/test2.dylib
# RUN: llvm-nm %t/test2.dylib | FileCheck %s
# CHECK-DAG: t _foo
# CHECK-DAG: d _bar

## If an archive has already been loaded without -load_hidden earlier in the command line,
## -load_hidden does not have an effect.
# RUN: %lld -dylib -lSystem %t/obj.o %t/libfoo.a -load_hidden %t/libfoo.a -o %t/test-regular-then-hidden.dylib
# RUN: llvm-nm %t/test-regular-then-hidden.dylib | FileCheck %s --check-prefix=REGULAR-THEN-HIDDEN
# REGULAR-THEN-HIDDEN-DAG: T _foo
# REGULAR-THEN-HIDDEN-DAG: D _bar

## If -load_hidden comes first, the symbols will have hidden visibility.
# RUN: %lld -dylib -lSystem %t/obj.o -load_hidden %t/libfoo.a %t/libfoo.a -o %t/test-hidden-then-regular.dylib
# RUN: llvm-nm %t/test-hidden-then-regular.dylib | FileCheck %s --check-prefix=HIDDEN-THEN-REGULAR
# HIDDEN-THEN-REGULAR-DAG: t _foo
# HIDDEN-THEN-REGULAR-DAG: d _bar

## If both -load_hidden and -force_load are specified, the earlier one will have an effect.
# RUN: %lld -dylib -lSystem %t/obj.o %t/libfoo.a -force_load %t/libbaz.a -load_hidden %t/libbaz.a -o %t/test-force-then-hidden.dylib
# RUN: llvm-nm %t/test-force-then-hidden.dylib | FileCheck %s --check-prefix=FORCE-THEN-HIDDEN
# FORCE-THEN-HIDDEN: T _baz
# RUN: %lld -dylib -lSystem %t/obj.o %t/libfoo.a -load_hidden %t/libbaz.a -force_load %t/libbaz.a -o %t/test-hidden-then-force.dylib
# RUN: llvm-nm %t/test-hidden-then-force.dylib | FileCheck %s --check-prefix=HIDDEN-THEN-FORCE
# HIDDEN-THEN-FORCE-NOT: _baz

## -load_hidden does not cause the library to be loaded eagerly.
# RUN: %lld -dylib -lSystem %t/obj.o -load_hidden %t/libfoo.a -load_hidden %t/libbaz.a -o %t/test-lazy.dylib
# RUN: llvm-nm %t/test-lazy.dylib | FileCheck %s --check-prefix=LAZY
# LAZY-NOT: _baz

## Specifying the same library twice is fine.
# RUN: %lld -dylib -lSystem %t/obj.o -load_hidden %t/libfoo.a -load_hidden %t/libfoo.a -o %t/test-twice.dylib
# RUN: llvm-nm %t/test-twice.dylib | FileCheck %s --check-prefix=TWICE
# TWICE-DAG: t _foo
# TWICE-DAG: d _bar

## -load_hidden causes the symbols to have "private external" visibility, so duplicate definitions
## are not allowed.
# RUN: not %lld -dylib -lSystem %t/obj.o %t/obj-bar.o -load_hidden %t/libfoo.a 2>&1 | FileCheck %s --check-prefix=DUP
# DUP: error: duplicate symbol: _bar

#--- archive-foo.s
.globl _foo
_foo:

.data
.globl _bar
_bar:

#--- archive-baz.s
.globl _baz
_baz:

#--- obj-bar.s
.data
.globl _bar
_bar:

#--- obj.s
.globl _test
_test:
    call _foo
