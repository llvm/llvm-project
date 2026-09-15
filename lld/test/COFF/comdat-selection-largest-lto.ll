; REQUIRES: x86

; Verify that definitions which disappear with a non-prevailing COMDAT are
; not mistaken for undefined references by the pre-LTO unresolved-symbol pass.

; RUN: split-file %s %t.dir
; RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
; RUN:   %t.dir/small.s -o %t.small.obj
; RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
; RUN:   %t.dir/large.s -o %t.large.obj
; RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
; RUN:   %t.dir/root.s -o %t.root.obj
; RUN: llvm-as %t.dir/dummy.ll -o %t.dummy.obj
; RUN: llvm-as %t.dir/losing-comdat.ll -o %t.losing-comdat.obj
; RUN: llvm-as %t.dir/undefined.ll -o %t.undefined.obj
; RUN: llvm-as %t.dir/weak-provider.ll -o %t.weak-provider.obj
; RUN: llvm-as %t.dir/strong-provider.ll -o %t.strong-provider.obj
; RUN: llvm-as %t.dir/common-provider.ll -o %t.common-provider.obj
; RUN: llvm-as %t.dir/deferred-comdat-provider.ll \
; RUN:   -o %t.deferred-comdat-provider.obj
; RUN: llvm-as %t.dir/mixed-leader.ll -o %t.mixed-leader.obj
; RUN: llvm-as %t.dir/mixed-child.ll -o %t.mixed-child.obj
; RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
; RUN:   %t.dir/mixed-native.s -o %t.mixed-native.obj
; RUN: opt -module-summary %t.dir/weak-provider.ll -o %t.weak-provider-thin.obj

; A native candidate which loses before it is read leaves a non-reference
; Undefined placeholder for its secondary symbols. The presence of unrelated
; bitcode must not turn that placeholder into a pre-LTO error.
; RUN: lld-link /dll /noentry /nodefaultlib /include:leader %t.large.obj \
; RUN:   %t.small.obj %t.dummy.obj /out:%t.native-loser.dll
; RUN: llvm-objdump -s %t.native-loser.dll | \
; RUN:   FileCheck --check-prefix=IMAGE --implicit-check-not=11111111 %s

; Replacing an already prevailing native candidate creates the same kind of
; non-reference placeholder for symbols unique to the old group.
; RUN: lld-link /dll /noentry /nodefaultlib /include:leader %t.small.obj \
; RUN:   %t.large.obj %t.dummy.obj /out:%t.native-replaced.dll
; RUN: llvm-objdump -s %t.native-replaced.dll | \
; RUN:   FileCheck --check-prefix=IMAGE --implicit-check-not=11111111 %s

; A weak bitcode definition suppressed by the smaller native group remains a
; valid provider if that group is later superseded.
; RUN: lld-link /dll /noentry /nodefaultlib /include:only_small \
; RUN:   %t.small.obj %t.weak-provider.obj %t.large.obj \
; RUN:   /out:%t.weak-provider.dll

; Bitcode providers registered before the provisional native definition are
; retained as well, for both weak and strong definitions.
; RUN: lld-link /dll /noentry /nodefaultlib /include:only_small \
; RUN:   %t.weak-provider.obj %t.small.obj %t.large.obj \
; RUN:   /out:%t.weak-provider-before.dll
; RUN: lld-link /dll /noentry /nodefaultlib /include:only_small \
; RUN:   %t.strong-provider.obj %t.small.obj %t.large.obj \
; RUN:   /out:%t.strong-provider-before.dll
; RUN: llvm-objdump --no-print-imm-hex -d %t.strong-provider-before.dll | \
; RUN:   FileCheck --check-prefix=STRONG %s

; A normal relocation from a regular object must retain and select a weak
; ThinLTO provider; /include is not required to make the reference visible.
; RUN: lld-link /entry:entry /subsystem:console /nodefaultlib \
; RUN:   %t.small.obj %t.weak-provider-thin.obj %t.large.obj %t.root.obj \
; RUN:   /out:%t.weak-provider-thin.exe

; Replaying providers uses normal weak/strong precedence rather than keeping
; the first DefinedRegular unconditionally.
; RUN: lld-link /dll /noentry /nodefaultlib /include:only_small \
; RUN:   %t.small.obj %t.weak-provider.obj %t.strong-provider.obj \
; RUN:   %t.large.obj /out:%t.strong-provider.dll
; RUN: llvm-objdump --no-print-imm-hex -d %t.strong-provider.dll | \
; RUN:   FileCheck --check-prefix=STRONG %s

; Replaying a common provider from bitcode must not try to read a COFF symbol
; record from the BitcodeFile.
; RUN: lld-link /dll /noentry /nodefaultlib /include:only_small \
; RUN:   %t.small.obj %t.common-provider.obj %t.large.obj \
; RUN:   /out:%t.common-provider.dll
; RUN: llvm-readobj --sections %t.common-provider.dll | \
; RUN:   FileCheck --check-prefix=COMMON %s

; Secondary definitions in a deferred bitcode COMDAT follow the eventual
; selection of their leader. They must neither leak from a rejected group nor
; disappear from a group selected after the native definition is replaced.
; RUN: not lld-link /force:multiple /dll /noentry /nodefaultlib \
; RUN:   /include:bitcode_child %t.small.obj %t.deferred-comdat-provider.obj \
; RUN:   /out:%t.rejected-bitcode-comdat.dll 2>&1 | \
; RUN:   FileCheck --check-prefix=BITCODE-REJECTED %s
; RUN: lld-link /dll /noentry /nodefaultlib /include:only_small \
; RUN:   /include:bitcode_child \
; RUN:   %t.small.obj %t.deferred-comdat-provider.obj %t.large.obj \
; RUN:   /out:%t.selected-bitcode-comdat.dll
; RUN: llvm-objdump -s %t.selected-bitcode-comdat.dll | \
; RUN:   FileCheck --check-prefix=BITCODE-SELECTED %s

; A symbol-table slot may contain both a deferred COMDAT leader and a child of
; another deferred COMDAT. Replay must revisit the slot after the child's
; parent has been selected, regardless of bitcode input order.
; RUN: lld-link /force:multiple /dll /noentry /nodefaultlib \
; RUN:   %t.mixed-native.obj %t.mixed-leader.obj %t.mixed-child.obj \
; RUN:   /out:%t.mixed-leader-first.dll 2>&1 | FileCheck --check-prefix=MIXED %s
; RUN: lld-link /force:multiple /dll /noentry /nodefaultlib \
; RUN:   %t.mixed-native.obj %t.mixed-child.obj %t.mixed-leader.obj \
; RUN:   /out:%t.mixed-child-first.dll 2>&1 | FileCheck --check-prefix=MIXED %s

; The same rule applies to secondary definitions in a losing bitcode COMDAT.
; RUN: lld-link /dll /noentry /nodefaultlib /include:leader %t.large.obj \
; RUN:   %t.losing-comdat.obj /out:%t.bitcode-loser.dll
; RUN: llvm-objdump -s %t.bitcode-loser.dll | \
; RUN:   FileCheck --check-prefix=IMAGE %s

; A genuine bitcode undefined reference must still be diagnosed before LTO.
; RUN: not lld-link /dll /noentry /nodefaultlib %t.large.obj \
; RUN:   %t.undefined.obj /out:%t.undefined.dll 2>&1 | \
; RUN:   FileCheck --check-prefix=UNDEFINED %s

; IMAGE: Contents of section .text:
; IMAGE: 44444444

; UNDEFINED: error: undefined symbol: genuinely_missing
; STRONG: movl $5678, %eax
; COMMON: Name: .data
; COMMON: VirtualSize: 0x4
; BITCODE-REJECTED: undefined symbol: bitcode_child
; BITCODE-SELECTED: 78563412
; MIXED-DAG: warning: duplicate symbol: mixed_parent
; MIXED-DAG: warning: duplicate symbol: mixed_role

;--- small.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .byte 0x11

        .globl only_small
only_small:
        .byte 0x11

;--- large.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .long 0x44444444

;--- dummy.ll
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define void @dummy() {
  ret void
}

;--- root.s
        .text
        .globl entry
entry:
        callq only_small
        retq

;--- losing-comdat.ll
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

$leader = comdat any

define linkonce_odr void @leader() comdat {
  ret void
}

define linkonce_odr void @only_bitcode() comdat($leader) {
  ret void
}

;--- undefined.ll
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define void @references_missing() {
  call void @genuinely_missing()
  ret void
}

declare void @genuinely_missing()

;--- weak-provider.ll
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define weak i32 @only_small() {
  ret i32 1234
}

;--- strong-provider.ll
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define i32 @only_small() {
  ret i32 5678
}

;--- common-provider.ll
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

@only_small = common global i32 0, align 4

;--- deferred-comdat-provider.ll
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

$only_small = comdat largest
; Keep the child before the leader: IR symbol order is independent of COMDAT
; membership, so deferred-group tracking must not rely on seeing the leader.
@bitcode_child = global i32 305419896, comdat($only_small)
@only_small = global [8 x i8] zeroinitializer, comdat($only_small)

;--- mixed-native.s
        .section .data$largest, "dw", largest, native_leader
        .globl native_leader
native_leader:
        .byte 0

        .globl mixed_role
mixed_role:
        .byte 0

        .globl mixed_parent
mixed_parent:
        .byte 0

;--- mixed-leader.ll
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

$mixed_role = comdat largest
@mixed_role = global [8 x i8] zeroinitializer, comdat

;--- mixed-child.ll
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

$mixed_parent = comdat largest
@mixed_role = global i32 42, comdat($mixed_parent)
@mixed_parent = global [8 x i8] zeroinitializer, comdat
