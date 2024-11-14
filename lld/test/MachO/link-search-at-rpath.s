# REQUIRES: x86

# RUN: rm -rf %t; split-file %s %t
# RUN: mkdir %t/subdir
# RUN: mkdir %t/subdir2

# RUN: llvm-mc -filetype obj -triple x86_64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype obj -triple x86_64-apple-darwin %t/bar.s -o %t/bar.o
# RUN: llvm-mc -filetype obj -triple x86_64-apple-darwin %t/main.s -o %t/main.o

# RUN: %lld -dylib -install_name @rpath/libfoo.dylib %t/foo.o -o %t/subdir/libfoo.dylib

# RUN: %lld -dylib -reexport_library %t/subdir/libfoo.dylib \
# RUN:     -rpath @loader_path/../foo \
# RUN:     -rpath @loader_path/../subdir \
# RUN:     -rpath @loader_path/../foo \
# RUN:     --no-warn-duplicate-rpath \
# RUN:     %t/bar.o -o %t/subdir2/libbar.dylib

# RUN: %lld -lSystem %t/main.o %t/subdir2/libbar.dylib -o %t/test
# RUN: %lld -dylib -lSystem %t/main.o %t/subdir2/libbar.dylib -o %t/libtest.dylib

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
