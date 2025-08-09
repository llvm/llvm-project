; REQUIRES: x86
;
; RUN: split-file %s %t.dir
;
; We need an input file to lld, so create one.
; RUN: llc -filetype=obj %t.dir/main.ll -o %t.obj

; RUN: lld-link -start-lib %t.obj /subsystem:console /force 2>&1 \
; RUN:     | FileCheck --check-prefix=MISSING_END %s
; MISSING_END: -start-lib without -end-lib

; RUN: not lld-link %t.obj -end-lib /subsystem:console /force 2>&1 \
; RUN:     | FileCheck --check-prefix=STRAY_END %s
; STRAY_END: stray -end-lib

; RUN: not lld-link -start-lib -start-lib %t.obj /subsystem:console /force 2>&1 \
; RUN:     | FileCheck --check-prefix=NESTED_START %s
; NESTED_START: nested -start-lib

; RUN: lld-link -start-lib %t.obj %S/Inputs/resource.res -end-lib /subsystem:console /force 2>&1 \
; RUN:     | FileCheck --check-prefix=WARN_RES %s
; WARN_RES: .res file provided between -start-lib/-end-lib will not be lazy

; RUN: lld-link -start-lib %t.obj %S/Inputs/ret42.lib -end-lib /subsystem:console /force 2>&1 \
; RUN:     | FileCheck --check-prefix=WARN_LIB %s
; WARN_LIB: .lib/.a file provided between -start-lib/-end-lib has no effect

; RUN: lld-link -start-lib %t.obj %S/Inputs/pdb-diff-cl.pdb -end-lib /subsystem:console /force 2>&1 \
; RUN:     | FileCheck --check-prefix=WARN_PDB %s
; WARN_PDB: .pdb file provided between -start-lib/-end-lib will not be lazy

; RUN: llvm-mc -filetype=obj -triple=x86_64-windows-gnu %t.dir/lib.s -o %t.lib.o
; RUN: lld-link -noentry -dll -def:%t.dir/lib.def %t.lib.o -out:%t.lib.dll -implib:%t.implib.lib
; RUN: lld-link -lldmingw -start-lib %t.lib.dll -end-lib /force 2>&1 \
; RUN:     | FileCheck --check-prefix=WARN_EXE %s
; WARN_EXE: .dll file provided between -start-lib/-end-lib will not be lazy

#--- main.ll
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define void @main() {
  ret void
}

#--- lib.s
.text
.global func1
func1:
  ret

#--- lib.def
NAME lib.dll
EXPORTS
    func1