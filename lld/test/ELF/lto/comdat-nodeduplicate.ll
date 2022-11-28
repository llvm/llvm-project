; REQUIRES: x86, shell

;; Keep __profd_foo in a nodeduplicate comdat, despite a comdat of the same name
;; in a previous object file.

;; Regular LTO

; RUN: rm -rf %t && split-file %s %t
; RUN: llvm-as %t/a.ll -o %t/a.bc
; RUN: llvm-as %t/b.ll -o %t/b.bc
; RUN: llvm-as %t/c.ll -o %t/c.bc

; RUN: ld.lld --save-temps -u foo %t/a.bc --start-lib %t/b.bc --end-lib -o %t/ab
; RUN: FileCheck %s --check-prefix=RESOL_AB < %t/ab.resolution.txt
; RUN: llvm-readelf -x .data %t/ab | FileCheck %s --check-prefix=DATA

; RESOL_AB: -r={{.*}}b.bc,__profc_foo,pl{{$}}

;; .data contains a.bc:data. b.bc:data and c.bc:data are discarded.
; DATA: 0x[[#%x,]] 01000000 00000000  ........

;; __profc_foo from c.bc is non-prevailing and thus discarded.
; RUN: ld.lld --save-temps -u foo -u c %t/a.bc --start-lib %t/b.bc %t/c.bc --end-lib -o %t/abc
; RUN: FileCheck %s --check-prefix=RESOL_ABC < %t/abc.resolution.txt
; RUN: llvm-readelf -x .data %t/abc | FileCheck %s --check-prefix=DATA

; RESOL_ABC: -r={{.*}}b.bc,__profc_foo,pl{{$}}
; RESOL_ABC: -r={{.*}}c.bc,__profc_foo,{{$}}

;; ThinLTO

; RUN: rm -rf %t && split-file %s %t
; RUN: opt --module-summary %t/a.ll -o %t/a.bc
; RUN: opt --module-summary %t/b.ll -o %t/b.bc
; RUN: opt --module-summary %t/c.ll -o %t/c.bc

; RUN: ld.lld --thinlto-index-only --save-temps -u foo %t/a.bc %t/b.bc -o %t/ab
; RUN: FileCheck %s --check-prefix=RESOL_AB < %t/ab.resolution.txt
; RUN: (llvm-dis < %t/b.bc && llvm-dis < %t/b.bc.thinlto.bc) | FileCheck %s --check-prefix=IR_AB
; RUN: ld.lld -u foo %t/a.bc %t/b.bc -o %t/ab
; RUN: llvm-readelf -x .data %t/ab | FileCheck %s --check-prefix=DATA

; RUN: ld.lld --thinlto-index-only --save-temps -u foo %t/a.bc --start-lib %t/b.bc --end-lib -o %t/ab
; RUN: FileCheck %s --check-prefix=RESOL_AB < %t/ab.resolution.txt
; RUN: (llvm-dis < %t/b.bc && llvm-dis < %t/b.bc.thinlto.bc) | FileCheck %s --check-prefix=IR_AB
; RUN: ld.lld -u foo %t/a.bc --start-lib %t/b.bc --end-lib -o %t/ab
; RUN: llvm-readelf -x .data %t/ab | FileCheck %s --check-prefix=DATA

; RUN: ld.lld --thinlto-index-only --save-temps -u foo -u c %t/a.bc --start-lib %t/b.bc %t/c.bc --end-lib -o %t/abc
; RUN: FileCheck %s --check-prefix=RESOL_ABC < %t/abc.resolution.txt
; RUN: (llvm-dis < %t/b.bc && llvm-dis < %t/b.bc.thinlto.bc) | FileCheck %s --check-prefix=IR_ABC
; RUN: ld.lld -u foo %t/a.bc --start-lib %t/b.bc %t/c.bc --end-lib -o %t/abc
; RUN: llvm-readelf -x .data %t/abc | FileCheck %s --check-prefix=DATA

; IR_AB-DAG: gv: (name: "__profd_foo", {{.*}} guid = [[PROFD:[0-9]+]]
; IR_AB-DAG: gv: (name: "__profc_foo", {{.*}} guid = [[PROFC:[0-9]+]]

;; Check extra attributes. b.bc:__profc_foo is prevailing, so it can be internalized.
; IR_AB-DAG: gv: (guid: [[PROFD]], summaries: (variable: (module: ^0, flags: (linkage: private, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 1, canAutoHide: 0), varFlags: (readonly: 0, writeonly: 0, constant: 0),
; IR_AB-DAG: gv: (guid: [[PROFC]], summaries: (variable: (module: ^0, flags: (linkage: internal, visibility: hidden, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0), varFlags: (readonly: 0, writeonly: 0, constant: 0))))

; IR_ABC-DAG: gv: (name: "__profd_foo", {{.*}} guid = [[PROFD:[0-9]+]]
; IR_ABC-DAG: gv: (name: "__profc_foo", {{.*}} guid = [[PROFC:[0-9]+]]

;; b.bc:__profc_foo prevails c.bc:__profc_foo, so it is exported and therefore not internalized.
; IR_ABC-DAG: gv: (guid: [[PROFD]], summaries: (variable: (module: ^0, flags: (linkage: private, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 1, canAutoHide: 0), varFlags: (readonly: 0, writeonly: 0, constant: 0),
; IR_ABC-DAG: gv: (guid: [[PROFC]], summaries: (variable: (module: ^0, flags: (linkage: weak, visibility: hidden, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0), varFlags: (readonly: 0, writeonly: 0, constant: 0))))

;--- a.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$__profc_foo = comdat nodeduplicate
@__profc_foo = private global i64 1, comdat, align 8
@__profd_foo = private global ptr @__profc_foo, comdat($__profc_foo), align 8

declare void @b()

define i64 @foo() {
  %v = load i64, ptr @__profc_foo
  %inc = add i64 1, %v
  store i64 %inc, ptr @__profc_foo
  ret i64 %inc
}

define void @_start() {
  call void @b()
  ret void
}

;--- b.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$__profc_foo = comdat nodeduplicate
@__profc_foo = weak hidden global i64 2, comdat, align 8
@__profd_foo = private global ptr @__profc_foo, comdat($__profc_foo)

define weak i64 @foo() {
  %v = load i64, ptr @__profc_foo
  %inc = add i64 1, %v
  store i64 %inc, ptr @__profc_foo
  ret i64 %inc
}

define void @b() {
  ret void
}

;--- c.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$__profc_foo = comdat nodeduplicate
@__profc_foo = weak hidden global i64 3, comdat, align 8
@__profd_foo = private global ptr @__profc_foo, comdat($__profc_foo)

define weak i64 @foo() {
  %v = load i64, ptr @__profc_foo
  %inc = add i64 1, %v
  store i64 %inc, ptr @__profc_foo
  ret i64 %inc
}

define void @c() {
  ret void
}
