# REQUIRES: x86
# UNSUPPORTED: system-windows
## FIXME: In principle this test should pass on Windows
# RUN: rm -rf %t; split-file %s %t

## This test verifies that we attempt to re-root absolute paths if -syslibroot
## is specified. Therefore we would like to be able to specify an absolute path
## without worrying that it may match an actual file on the system outside the
## syslibroot. `chroot` would do the job but isn't cross-platform, so I've used
## this %t/%:t hack instead.
# RUN: mkdir -p %t/%:t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/bar.s -o %t/bar.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o

## bar.a is under %t/%:t, and so verifies that rerooting happens. foo.a isn't,
## and therefore verifies that we still fall back to the original path if no
## file exists at the rerooted path.
# RUN: llvm-ar rcs %t/foo.a %t/foo.o
# RUN: %lld -dylib %t/foo.o -o %t/libfoo.dylib
# RUN: llvm-ar rcs %t/%:t/bar.a %t/bar.o
# RUN: %lld -dylib %t/bar.o -o %t/%:t/libbar.dylib

## Test our various file-loading flags to make sure all bases are covered.
# RUN: %lld --reproduce %t/repro1.tar -lSystem -syslibroot %t %t/foo.a %t/bar.a %t/test.o -o /dev/null -t | FileCheck %s -DDIR="%t/%:t"
# RUN: tar xf %t/repro1.tar -C %t
# RUN: cd %t/repro1; %no-arg-lld @response.txt | FileCheck %s -DDIR="%:t/%:t"

# RUN: %lld --reproduce %t/repro2.tar -lSystem -syslibroot %t -force_load %t/foo.a -force_load %t/bar.a %t/test.o -o /dev/null -t | FileCheck %s -DDIR="%t/%:t"
# RUN: tar xf %t/repro2.tar -C %t
# RUN: cd %t/repro2; %no-arg-lld @response.txt | FileCheck %s -DDIR="%:t/%:t"

# RUN: %lld --reproduce %t/repro3.tar -lSystem -syslibroot %t %t/libfoo.dylib %t/libbar.dylib %t/test.o -o /dev/null -t | FileCheck %s -DDIR="%t/%:t"
# RUN: tar xf %t/repro3.tar -C %t
# RUN: cd %t/repro3; %no-arg-lld @response.txt | FileCheck %s -DDIR="%:t/%:t"

# RUN: %lld --reproduce %t/repro4.tar -lSystem -syslibroot %t -weak_library %t/libfoo.dylib -weak_library %t/libbar.dylib %t/test.o -o /dev/null -t | FileCheck %s -DDIR="%t/%:t"
# RUN: tar xf %t/repro4.tar -C %t
# RUN: cd %t/repro4; %no-arg-lld @response.txt | FileCheck %s -DDIR="%:t/%:t"

# RUN: %lld --reproduce %t/repro7.tar -lSystem -syslibroot %t -force_load %t/foo.a -force_load %t/bar.a %t/test.o -o /dev/null -t | FileCheck %s -DDIR="%t/%:t"
# RUN: tar xf %t/repro7.tar -C %t
# RUN: cd %t/repro7; %no-arg-lld @response.txt | FileCheck %s -DDIR="%:t/%:t"

# RUN: echo "%t/libfoo.dylib" > %t/filelist
# RUN: echo "%t/libbar.dylib" >> %t/filelist
# RUN: %lld --reproduce %t/repro5.tar -lSystem -syslibroot %t -filelist %t/filelist %t/test.o -o /dev/null -t | FileCheck %s -DDIR="%t/%:t"
# RUN: tar xf %t/repro5.tar -C %t
# RUN: cd %t/repro5; %no-arg-lld @response.txt | FileCheck %s -DDIR="%:t/%:t"

## The {{^}} ensures that we only match relative paths if DIR is relative.
# CHECK: {{^}}[[DIR]]/{{(lib)?}}bar

## Paths to object files don't get rerooted.
# RUN: mv %t/bar.o %t/%:t/bar.o
# RUN: not %lld -lSystem -syslibroot %t %t/foo.o %t/bar.o %t/test.o -o \
# RUN:   /dev/null 2>&1 | FileCheck %s --check-prefix=OBJ
# OBJ: error: cannot open {{.*[\\/]}}bar.o: {{[Nn]}}o such file or directory

## Now create a "decoy" libfoo.dylib under %t/%:t to demonstrate that the
## rerooted path takes precedence over the original path. We will get an
## undefined symbol error since we aren't loading %t/libfoo.dylib.
# RUN: cp %t/%:t/libbar.dylib %t/%:t/libfoo.dylib
# RUN: not %lld --reproduce %t/repro6.tar -lSystem -syslibroot %t %t/libfoo.dylib %t/libbar.dylib %t/test.o \
# RUN:   -o /dev/null -t 2> %t/error | FileCheck %s -DDIR="%t/%:t" --check-prefix=FOO
# RUN: FileCheck %s --check-prefix=UNDEF < %t/error

# RUN: tar xf %t/repro6.tar -C %t
# RUN: cd %t/repro6; not %no-arg-lld @response.txt 2> %t/error | FileCheck %s -DDIR="%:t/%:t" --check-prefix=FOO
# RUN: FileCheck %s --check-prefix=UNDEF < %t/error

# FOO: [[DIR]]/libfoo.dylib
# UNDEF: error: undefined symbol: _foo

#--- foo.s
.globl _foo
_foo:

#--- bar.s
.globl _bar
_bar:

#--- test.s
.text
.globl _main

_main:
  callq _foo
  callq _bar
  ret
