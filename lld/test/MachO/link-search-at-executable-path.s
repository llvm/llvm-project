# REQUIRES: x86

# RUN: rm -rf %t; split-file %s %t
# RUN: mkdir %t/subdir

# RUN: llvm-mc -filetype obj -triple x86_64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype obj -triple x86_64-apple-darwin %t/bar.s -o %t/bar.o
# RUN: llvm-mc -filetype obj -triple x86_64-apple-darwin %t/main.s -o %t/main.o

# RUN: %lld -dylib -install_name @executable_path/libfoo.dylib %t/foo.o -o %t/subdir/libfoo.dylib

# RUN: %lld -dylib -reexport_library %t/subdir/libfoo.dylib %t/bar.o -o %t/libbar.dylib

## When linking executables, @executable_path/ in install_name should be replaced
## by the path of the executable.
# RUN: %lld -lSystem %t/main.o %t/libbar.dylib -o %t/subdir/test

## This doesn't work for non-executables.
# RUN: not %lld -dylib -lSystem %t/main.o %t/libbar.dylib -o %t/subdir/libtest.dylib 2>&1 | FileCheck --check-prefix=ERR %s

## It also doesn't help if the needed reexport isn't next to the library.
# RUN: not %lld -lSystem %t/main.o %t/libbar.dylib -o %t/test 2>&1 | FileCheck --check-prefix=ERR %s
# ERR: error: {{.*}}libbar.dylib: unable to locate re-export with install name @executable_path/libfoo.dylib

#--- foo.s
.globl _foo
_foo:
  retq

#--- bar.s
.globl _bar
_bar:
  retq

#--- main.s
.section __TEXT,__text
.global _main
_main:
  callq _foo
  callq _bar
  ret
