# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t

## This test demonstrates that when an archive file is fetched, its symbols
## always override any conflicting dylib symbols, regardless of any weak
## definition flags.

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/libfoo.s -o %t/libfoo.o
# RUN: %lld -dylib -install_name @executable_path/libfoo.dylib %t/libfoo.o -o %t/libfoo.dylib

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/weakfoo.s -o %t/weakfoo.o

# RUN: llvm-ar --format=darwin rcs %t/foo.a %t/foo.o
# RUN: llvm-ar --format=darwin rcs %t/weakfoo.a %t/weakfoo.o

# PREFER-WEAK-OBJECT: O __TEXT,weak _foo
# PREFER-NONWEAK-OBJECT: O __TEXT,nonweak _foo

# RUN: %lld -lSystem -o %t/nonweak-dylib-weak-ar -L%t -lfoo %t/weakfoo.a %t/test.o
# RUN: llvm-objdump --macho --lazy-bind --syms %t/nonweak-dylib-weak-ar | FileCheck %s --check-prefix=PREFER-WEAK-OBJECT
# RUN: %lld -lSystem -o %t/weak-ar-nonweak-dylib -L%t %t/weakfoo.a -lfoo %t/test.o
# RUN: llvm-objdump --macho --lazy-bind --syms %t/weak-ar-nonweak-dylib | FileCheck %s --check-prefix=PREFER-WEAK-OBJECT

# RUN: %lld -lSystem -o %t/weak-obj-nonweak-dylib -L%t %t/weakfoo.o -lfoo %t/test.o
# RUN: llvm-objdump --macho --lazy-bind --syms %t/weak-obj-nonweak-dylib | FileCheck %s --check-prefix=PREFER-WEAK-OBJECT
# RUN: %lld -lSystem -o %t/nonweak-dylib-weak-obj -L%t -lfoo %t/weakfoo.o %t/test.o
# RUN: llvm-objdump --macho --lazy-bind --syms %t/nonweak-dylib-weak-obj | FileCheck %s --check-prefix=PREFER-WEAK-OBJECT

#--- libfoo.s
.globl _foo
_foo:

#--- foo.s
.globl _foo, _bar
.section __TEXT,nonweak
_bar:
_foo:

#--- weakfoo.s
.globl _foo, _bar
.weak_definition _foo
.section __TEXT,weak
_bar:
_foo:

#--- test.s
.globl _main
_main:
  callq _foo
  callq _bar
  ret
