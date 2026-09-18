; REQUIRES: x86
;
; Exercise deferred COMDAT state in two driver epochs. Native inputs first
; create and drain deferred providers. LTO then emits another native object
; containing a replaceable COMDAT, which must be processed with freshly valid
; provider IDs and empty per-epoch work queues.
;
; RUN: split-file %s %t.dir
; RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
; RUN:   %t.dir/pre-small.s -o %t.pre-small.obj
; RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
; RUN:   %t.dir/pre-large.s -o %t.pre-large.obj
; RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
; RUN:   %t.dir/pre-provider.s -o %t.pre-provider.obj
; RUN: llvm-lib -machine:amd64 -out:%t.pre-provider.lib %t.pre-provider.obj
; RUN: llvm-as %t.dir/epoch-large.ll -o %t.epoch-large.obj
; RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
; RUN:   %t.dir/root.s -o %t.root.obj
;
; RUN: lld-link /entry:entry /subsystem:console /nodefaultlib /opt:noref \
; RUN:   %t.pre-small.obj %t.pre-provider.lib %t.pre-large.obj \
; RUN:   %t.epoch-large.obj %t.root.obj \
; RUN:   /out:%t.exe /map:%t.map
; RUN: FileCheck %s --check-prefix=MAP < %t.map
; RUN: llvm-objdump -s %t.exe | FileCheck %s --check-prefix=IMAGE
;
; MAP: pre_only{{.*}}pre-provider.obj
; IMAGE: 88776655
; IMAGE-NOT: 11111111
;
;--- pre-small.s
        .section .text$pre, "xr", largest, pre_leader
        .globl pre_leader
pre_leader:
        .byte 0x11
        .globl pre_only
pre_only:
        .long 0x11111111
;
;--- pre-large.s
        .section .text$pre, "xr", largest, pre_leader
        .globl pre_leader
pre_leader:
        .space 32, 0x44
;
;--- pre-provider.s
        .section .rdata$provider, "dr"
        .globl pre_only
pre_only:
        .long 0x22222222
;
;--- epoch-large.ll
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

$epoch_leader = comdat largest

@epoch_leader = global [64 x i8] zeroinitializer, comdat
@epoch_child = global i32 1432778632, comdat($epoch_leader)
;
;--- root.s
        .text
        .globl entry
entry:
        movl pre_only(%rip), %eax
        leaq epoch_leader(%rip), %rcx
        movl epoch_child(%rip), %edx
        retq
