; REQUIRES: arm-registered-target
; RUN: mkdir -p %t
; RUN: opt -module-summary -o %t/foo.o %s
; RUN: opt -module-summary -o %t/bar.o %S/Inputs/linker-diagnostic1.ll
; RUN: llvm-lto2 run --thinlto-distributed-indexes -r %t/foo.o,foo,plx -r %t/bar.o,bar,plx \
; RUN:   -r %t/bar.o,foo, -o %t/foobar.so %t/foo.o %t/bar.o
; RUN: %clang -c -o %t/lto.bar.o --target=armv4-none-unknown-eabi -O2 \
; RUN:   -fthinlto-index=%t/bar.o.thinlto.bc %t/bar.o -Wno-override-module 2>&1 | FileCheck %s

; CHECK: linking module '{{.*}}/bar.o': Linking two modules of different target triples: '{{.*}}/foo.o' is 'thumbv6-unknown-linux-gnueabihf' whereas '{{.*}}/bar.o' is 'armv4-none-unknown-eabi'

target triple = "thumbv6-unknown-linux-gnueabihf"
target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

define i32 @foo(i32 %x) {
  %1 = add i32 %x, 1
  ret i32 %1
}
