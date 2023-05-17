; REQUIRES: x86

; RUN: echo -e ".global variable\n.global DllMainCRTStartup\n.text\nDllMainCRTStartup:\nret\n.data\nvariable:\n.long 42" > %t-lib.s
; RUN: llvm-mc -triple=x86_64-windows-gnu %t-lib.s -filetype=obj -o %t-lib.obj
; RUN: lld-link -out:%t-lib.dll -dll -entry:DllMainCRTStartup %t-lib.obj -lldmingw -implib:%t-lib.lib

; RUN: llvm-as -o %t.obj %s
; RUN: lld-link -lldmingw -out:%t.exe -entry:entry %t.obj %t-lib.lib

; RUN: llvm-readobj --coff-imports %t.exe | FileCheck -check-prefix=IMPORTS %s

; IMPORTS: Import {
; IMPORTS-NEXT: Name: autoimport-lto.ll.tmp-lib.dll
; IMPORTS-NEXT: ImportLookupTableRVA:
; IMPORTS-NEXT: ImportAddressTableRVA:
; IMPORTS-NEXT: Symbol: variable (0)
; IMPORTS-NEXT: }

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-w64-windows-gnu"

@variable = external global i32

define i32 @entry() {
entry:
  %0 = load i32, ptr @variable
  ret i32 %0
}
