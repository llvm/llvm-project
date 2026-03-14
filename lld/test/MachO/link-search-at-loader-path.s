# REQUIRES: x86

# RUN: rm -rf %t; split-file %s %t
# RUN: mkdir %t/subdir

# RUN: llvm-mc -filetype obj -triple x86_64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype obj -triple x86_64-apple-darwin %t/bar.s -o %t/bar.o
# RUN: llvm-mc -filetype obj -triple x86_64-apple-darwin %t/main.s -o %t/main.o

# RUN: %lld -dylib -install_name @loader_path/libfoo.dylib %t/foo.o -o %t/subdir/libfoo.dylib

## Test that @loader_path is replaced by the actual path, not by install_name.
# RUN: %lld -dylib -reexport_library %t/subdir/libfoo.dylib -install_name /tmp/libbar.dylib %t/bar.o -o %t/subdir/libbar.dylib

# RUN: %lld -lSystem %t/main.o %t/subdir/libbar.dylib -o %t/test
# RUN: %lld -dylib -lSystem %t/main.o %t/subdir/libbar.dylib -o %t/libtest.dylib

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
