# REQUIRES: x86
# RUN: rm -rf %t
# RUN: split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/foo.o %t/foo.s
# RUN: %lld -dylib -o %t/libfoo.dylib %t/foo.o

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/bar.o %t/bar.s
# RUN: llvm-ar csr  %t/bar.a %t/bar.o

# RUN: llvm-as %t/baz.ll -o %t/baz.o

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/main.o %t/main.s

# RUN: %lld %t/main.o %t/baz.o %t/bar.a %t/libfoo.dylib -lSystem -o /dev/null -t | FileCheck -DPATH='%t' %s

# CHECK-DAG: bar.a(bar.o)
# CHECK-DAG: [[PATH]]/main.o
# CHECK-DAG: [[PATH]]/baz.o
# CHECK-DAG: [[PATH]]/libfoo.dylib
# CHECK-DAG: {{.*}}/usr/lib{{[/\\]}}libSystem.tbd

#--- foo.s
.globl __Z3foo
__Z3foo:
  ret

#--- bar.s
.globl _bar
_bar:
  callq __Z3foo
  ret

#--- baz.ll

target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @baz() {
  ret void
}

#--- main.s
.globl _main
_main:
  callq _bar
  callq __Z3foo
  callq _baz
  ret
