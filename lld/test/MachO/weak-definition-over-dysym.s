# REQUIRES: x86
# RUN: mkdir -p %t

## This test demonstrates that when an archive file is fetched, its symbols
## always override any conflicting dylib symbols, regardless of any weak
## definition flags.

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/test.o
# RUN: echo ".globl _foo; _foo:" | llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/libfoo.o
# RUN: lld -flavor darwinnew -dylib -install_name \
# RUN:   @executable_path/libfoo.dylib %t/libfoo.o -o %t/libfoo.dylib

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/test.o
# RUN: echo ".globl _foo, _bar; .section __TEXT,nonweak; _bar: _foo:" | llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/foo.o
# RUN: echo ".globl _foo, _bar; .weak_definition _foo; .section __TEXT,weak; _bar: _foo:" | llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/weakfoo.o

# RUN: rm -f %t/foo.a
# RUN: llvm-ar --format=darwin rcs %t/foo.a %t/foo.o
# RUN: rm -f %t/weakfoo.a
# RUN: llvm-ar --format=darwin rcs %t/weakfoo.a %t/weakfoo.o

# PREFER-WEAK-OBJECT: O __TEXT,weak _foo
# PREFER-NONWEAK-OBJECT: O __TEXT,nonweak _foo

# RUN: lld -flavor darwinnew -L%S/Inputs/MacOSX.sdk/usr/lib -lSystem -o %t/nonweak-dylib-weak-ar -Z -L%t -lfoo %t/weakfoo.a %t/test.o
# RUN: llvm-objdump --macho --lazy-bind --syms %t/nonweak-dylib-weak-ar | FileCheck %s --check-prefix=PREFER-WEAK-OBJECT
# RUN: lld -flavor darwinnew -L%S/Inputs/MacOSX.sdk/usr/lib -lSystem -o %t/weak-ar-nonweak-dylib -Z -L%t %t/weakfoo.a -lfoo %t/test.o
# RUN: llvm-objdump --macho --lazy-bind --syms %t/weak-ar-nonweak-dylib | FileCheck %s --check-prefix=PREFER-WEAK-OBJECT

# RUN: lld -flavor darwinnew -L%S/Inputs/MacOSX.sdk/usr/lib -lSystem -o %t/weak-obj-nonweak-dylib -Z -L%t %t/weakfoo.o -lfoo %t/test.o
# RUN: llvm-objdump --macho --lazy-bind --syms %t/weak-obj-nonweak-dylib | FileCheck %s --check-prefix=PREFER-WEAK-OBJECT
# RUN: lld -flavor darwinnew -L%S/Inputs/MacOSX.sdk/usr/lib -lSystem -o %t/nonweak-dylib-weak-obj -Z -L%t -lfoo %t/weakfoo.o %t/test.o
# RUN: llvm-objdump --macho --lazy-bind --syms %t/nonweak-dylib-weak-obj | FileCheck %s --check-prefix=PREFER-WEAK-OBJECT

.globl _main
_main:
  callq _foo
  callq _bar
  ret
