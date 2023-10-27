; REQUIRES: x86

; RUN: llvm-as %s -o %t.obj
; RUN: lld-link -opt:lldlto=0 %t.obj -dll -noentry -out:%t.dll -mllvm:-debug-pass=Structure 2>&1 | FileCheck --check-prefix=OPT %s
; RUN: lld-link -opt:lldltocgo=0 %t.obj -dll -noentry -out:%t.dll -mllvm:-debug-pass=Structure 2>&1 | FileCheck --check-prefix=NOOPT %s
; RUN: lld-link -opt:lldlto=3 %t.obj -dll -noentry -out:%t.dll -mllvm:-debug-pass=Structure 2>&1 | FileCheck --check-prefix=OPT %s
; RUN: lld-link -opt:lldltocgo=3 %t.obj -dll -noentry -out:%t.dll -mllvm:-debug-pass=Structure 2>&1 | FileCheck --check-prefix=OPT %s
; RUN: lld-link -opt:lldlto=0 -opt:lldltocgo=0 %t.obj -dll -noentry -out:%t.dll -mllvm:-debug-pass=Structure 2>&1 | FileCheck --check-prefix=NOOPT %s
; RUN: lld-link -opt:lldlto=3 -opt:lldltocgo=0 %t.obj -dll -noentry -out:%t.dll -mllvm:-debug-pass=Structure 2>&1 | FileCheck --check-prefix=NOOPT %s
; RUN: lld-link -opt:lldlto=0 -opt:lldltocgo=3 %t.obj -dll -noentry -out:%t.dll -mllvm:-debug-pass=Structure 2>&1 | FileCheck --check-prefix=OPT %s
; RUN: lld-link -opt:lldlto=3 -opt:lldltocgo=3 %t.obj -dll -noentry -out:%t.dll -mllvm:-debug-pass=Structure 2>&1 | FileCheck --check-prefix=OPT %s
; RUN: lld-link -opt:lldlto=0 -opt:lldltocgo=0 -opt:lldltocgo=2 %t.obj -dll -noentry -out:%t.dll -mllvm:-debug-pass=Structure 2>&1 | FileCheck --check-prefix=OPT %s
; RUN: not lld-link -opt:lldlto=4 %t.obj -dll -noentry -out:%t.dll -mllvm:-debug-pass=Structure 2>&1 | FileCheck --check-prefix=ERROR-O4 %s
; RUN: not lld-link -opt:lldltocgo=4 %t.obj -dll -noentry -out:%t.dll -mllvm:-debug-pass=Structure 2>&1 | FileCheck --check-prefix=ERROR-CGO4 %s
; RUN: not lld-link -opt:lldlto=4 -opt:lldltocgo=4 %t.obj -dll -noentry -out:%t.dll -mllvm:-debug-pass=Structure 2>&1 | FileCheck --check-prefixes=ERROR-O4,ERROR-CGO4 %s

; NOOPT: Fast Register Allocator
; OPT: Greedy Register Allocator
; ERROR-O4: lld-link: error: /opt:lldlto: invalid optimization level: 4
; ERROR-CGO4: lld-link: error: /opt:lldltocgo: invalid codegen optimization level: 4

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define void @_start() {
entry:
  ret void
}
