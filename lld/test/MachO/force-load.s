# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/archive-foo.s -o %t/archive-foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/archive-baz.s -o %t/archive-baz.o
# RUN: llvm-ar rcs %t/foo.a %t/archive-foo.o
# RUN: llvm-ar rcs %t/baz.a %t/archive-baz.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o

# RUN: %lld -lSystem -force_load %t/foo.a %t/foo.o %t/test.o -o %t/test-force-load-first
# FORCE-LOAD-FIRST:  __TEXT,archive _foo
# RUN: llvm-objdump --syms %t/test-force-load-first | FileCheck %s --check-prefix=FORCE-LOAD-FIRST

# RUN: %lld %t/foo.o -lSystem -force_load %t/foo.a %t/test.o -o %t/test-force-load-second
# RUN: llvm-objdump --syms %t/test-force-load-second | FileCheck %s --check-prefix=FORCE-LOAD-SECOND
# FORCE-LOAD-SECOND: __TEXT,obj _foo

## If an archive has already been loaded w/o -force_load earlier in the command
## line, a later -force_load argument will not have an effect.
# RUN: %lld -lSystem %t/foo.a -force_load %t/foo.a %t/test.o -o %t/test-regular-then-force
# RUN: llvm-objdump --syms %t/test-regular-then-force | FileCheck %s --check-prefix=REGULAR-THEN-FORCE
# REGULAR-THEN-FORCE-NOT: _foo
## If the -force_load comes first, then the second load will just be a no-op.
# RUN: %lld -lSystem -force_load %t/foo.a %t/foo.a %t/test.o -o %t/test-force-then-regular
# RUN: llvm-objdump --syms %t/test-force-then-regular | FileCheck %s --check-prefix=FORCE-THEN-REGULAR
# FORCE-THEN-REGULAR: _foo

## Force-loading the same path twice is fine
# RUN: %lld -lSystem %t/foo.o -force_load %t/foo.a -force_load %t/foo.a %t/test.o -o /dev/null

## Note that we do not call realpath() before dedup'ing the force-load
## arguments, so this is an error.
# RUN: cd %t; not %lld -lSystem %t/foo.o -force_load %t/foo.a -force_load foo.a \
# RUN:   %t/test.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=DUP

# DUP: error: duplicate symbol: _bar

## Force-loading two different paths w/o conflicting symbols is fine
# RUN: %lld -lSystem -force_load %t/foo.a -force_load %t/baz.a %t/test.o -o %t/test-two-force-loads
# RUN: llvm-objdump --syms %t/test-two-force-loads | FileCheck %s --check-prefix=TWICE
# TWICE-DAG: __TEXT,archive _foo
# TWICE-DAG: __TEXT,archive _bar
# TWICE-DAG: __TEXT,archive _baz

## Loading an empty-archive should not crash.
# RUN: llvm-ar --format=darwin rcs %t/libEmpty.a
# RUN: %lld -lSystem %t/test.o -force_load %t/libEmpty.a -o %t/loadEmpty.out
# RUN: llvm-objdump --macho --syms %t/loadEmpty.out | FileCheck %s --check-prefix=EMPTY
# EMPTY: F __TEXT,__text _main

#--- archive-foo.s
.section __TEXT,archive
.globl _foo, _bar
.weak_definition _foo
_foo:
_bar:

#--- archive-baz.s
.section __TEXT,archive
.globl _baz
_baz:

#--- foo.s
.section __TEXT,obj
.globl _foo
.weak_definition _foo
_foo:

#--- test.s
.globl _main
_main:
  ret
