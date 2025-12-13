# REQUIRES: x86
# Windows does not support rpath
# UNSUPPORTED: system-windows

# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype obj -triple x86_64-apple-darwin %t/bar.s -o %t/bar.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o

## 1. Test symlink with reexport to @rpath and rpath pointing to @loader_path.
## The @loader_path-relative path is looked after resolving the symlink.

# RUN: mkdir -p %t/Foo1.framework/Versions/A
# RUN: %lld -dylib -install_name @rpath/libbar1.dylib %t/bar.o -o %t/Foo1.framework/Versions/A/libbar1.dylib

# RUN: %lld -dylib -install_name %t/Foo1.framework/Versions/A/Foo1 %t/foo.o \
# RUN:     -reexport_library %t/Foo1.framework/Versions/A/libbar1.dylib \
# RUN:     -rpath @loader_path/. \
# RUN:     -o %t/Foo1.framework/Versions/A/Foo1
# RUN: ln -sf A %t/Foo1.framework/Versions/Current
# RUN: ln -sf Versions/Current/Foo1 %t/Foo1.framework/Foo1

# RUN: %lld -lSystem -F%t -framework Foo1 %t/test.o -o %t/test1

## 2. Test symlink with reexport to @loader_path-relative path directly.
## The @loader_path-relative path is looked after resolving the symlink.
## ld64 gets this wrong -- it calls realpath() but ignores the result.
## (ld64-609, Options.cpp, Options::findFile(), "@loader_path/" handling.)

# RUN: mkdir -p %t/Foo2.framework/Versions/A
# RUN: %lld -dylib -install_name @loader_path/libbar2.dylib %t/bar.o -o %t/Foo2.framework/Versions/A/libbar2.dylib

# RUN: %lld -dylib -install_name %t/Foo2.framework/Versions/A/Foo2 %t/foo.o \
# RUN:     -reexport_library %t/Foo2.framework/Versions/A/libbar2.dylib \
# RUN:     -o %t/Foo2.framework/Versions/A/Foo2
# RUN: ln -sf A %t/Foo2.framework/Versions/Current
# RUN: ln -sf Versions/Current/Foo2 %t/Foo2.framework/Foo2

# RUN: %lld -lSystem -F%t -framework Foo2 %t/test.o -o %t/test2

#--- foo.s
.globl _foo
_foo:
  ret

#--- bar.s
.globl _bar
_bar:
  ret

#--- test.s
.globl _main
.text
_main:
  callq _foo
  callq _bar
  ret

