# REQUIRES: x86
# RUN: mkdir -p %t
# RUN: echo '.globl _foo, _bar, _baz; _foo: _bar: _baz:' | \
# RUN:   llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/libresolution.o
# RUN: %lld -dylib -install_name \
# RUN:   @executable_path/libresolution.dylib %t/libresolution.o -o %t/libresolution.dylib
# RUN: %lld -dylib -install_name \
# RUN:   @executable_path/libresolution2.dylib %t/libresolution.o -o %t/libresolution2.dylib
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/resolution.o

## Check that we select the symbol defined in the first dylib passed on the
## command line.
# RUN: %lld -o %t/dylib-first -L%t -lresolution -lresolution2 %t/resolution.o
# RUN: llvm-objdump --macho --bind %t/dylib-first | FileCheck %s --check-prefix=DYLIB-FIRST
# DYLIB-FIRST:     libresolution _foo

# RUN: %lld -o %t/dylib2-first -L%t -lresolution2 -lresolution %t/resolution.o
# RUN: llvm-objdump --macho --bind %t/dylib2-first | FileCheck %s --check-prefix=DYLIB2-FIRST
# DYLIB2-FIRST: libresolution2 _foo

## Also check that defined symbols take precedence over dylib symbols.
# DYLIB-FIRST-NOT: libresolution _bar
# DYLIB-FIRST-NOT: libresolution _baz

## Check that we pick the dylib symbol over the undefined symbol in the object
## file, even if the object file appears first on the command line.
# RUN: %lld -o %t/obj-first -L%t %t/resolution.o -lresolution
# RUN: llvm-objdump --macho --bind %t/obj-first | FileCheck %s --check-prefix=OBJ-FIRST
# OBJ-FIRST: libresolution _foo
## But defined symbols should still take precedence.
# OBJ-FIRST-NOT: libresolution _bar
# OBJ-FIRST-NOT: libresolution _baz

.globl _main, _bar
# Global defined symbol
_bar:
# Local defined symbol
_baz:

_main:
  movq _foo@GOTPCREL(%rip), %rsi
  movq _bar@GOTPCREL(%rip), %rsi
  movq _baz@GOTPCREL(%rip), %rsi
  ret
