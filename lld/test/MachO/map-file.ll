; REQUIRES: x86
; RUN: rm -rf %t; split-file %s %t

;; Verify that we map symbols to their original bitcode input files, not the
;; intermediate object files. Also verify that we don't emit these intermediate
;; object files if no symbol needs to reference them. (Intermediate object files
;; may still be referenced if they contain compiler-synthesized symbols, such
;; as outlined functions. TODO: Test this case.)

; RUN: opt -module-summary %t/foo.ll -o %t/foo.thinlto.o
; RUN: opt -module-summary %t/bar.ll -o %t/bar.thinlto.o

; RUN: %lld -dylib %t/foo.thinlto.o %t/bar.thinlto.o -o %t/foobar.thinlto -map \
; RUN:   %t/foobar.thinlto.map
; RUN: FileCheck %s --check-prefix=FOOBAR < %t/foobar.thinlto.map

; RUN: %lld -dylib %t/bar.thinlto.o %t/foo.thinlto.o -o %t/barfoo.thinlto -map \
; RUN:   %t/barfoo.thinlto.map
; RUN: FileCheck %s --check-prefix=BARFOO < %t/barfoo.thinlto.map

; RUN: llvm-as %t/foo.ll -o %t/foo.o
; RUN: llvm-as %t/bar.ll -o %t/bar.o
; RUN: %lld -dylib %t/foo.o %t/bar.o -o %t/foobar -map %t/foobar.map
; RUN: FileCheck %s --check-prefix=LTO < %t/foobar.map

; FOOBAR:      # Path: {{.*}}{{/|\\}}map-file.ll.tmp/foobar.thinlto
; FOOBAR-NEXT: # Arch: x86_64
; FOOBAR-NEXT: # Object files:
; FOOBAR-NEXT: [  0] linker synthesized
; FOOBAR-NEXT: [  1] {{.*}}{{/|\\}}usr/lib{{/|\\}}libSystem.tbd{{$}}
; FOOBAR-NEXT: [  2] {{.*}}{{/|\\}}map-file.ll.tmp/foo.thinlto.o{{$}}
; FOOBAR-NEXT: [  3] {{.*}}{{/|\\}}map-file.ll.tmp/bar.thinlto.o{{$}}
; FOOBAR-NEXT: # Sections:
; FOOBAR:      # Symbols:
; FOOBAR-NEXT: # Address        Size             File  Name
; FOOBAR-NEXT: 0x{{[0-9A-F]+}}  0x{{[0-9A-F]+}}  [  2] _foo
; FOOBAR-NEXT: 0x{{[0-9A-F]+}}  0x{{[0-9A-F]+}}  [  3] _bar
; FOOBAR-NEXT: 0x{{[0-9A-F]+}}  0x{{[0-9A-F]+}}  [  3] _maybe_weak

; BARFOO:      # Path: {{.*}}{{/|\\}}map-file.ll.tmp/barfoo.thinlto
; BARFOO-NEXT: # Arch: x86_64
; BARFOO-NEXT: # Object files:
; BARFOO-NEXT: [  0] linker synthesized
; BARFOO-NEXT: [  1] {{.*}}{{/|\\}}usr/lib{{/|\\}}libSystem.tbd
; BARFOO-NEXT: [  2] {{.*}}{{/|\\}}map-file.ll.tmp/bar.thinlto.o
; BARFOO-NEXT: [  3] {{.*}}{{/|\\}}map-file.ll.tmp/foo.thinlto.o
; BARFOO-NEXT: # Sections:
; BARFOO:      # Symbols:
; BARFOO-NEXT: # Address        Size             File  Name
; BARFOO-NEXT: 0x{{[0-9A-F]+}}  0x{{[0-9A-F]+}}  [  2] _bar
; BARFOO-NEXT: 0x{{[0-9A-F]+}}  0x{{[0-9A-F]+}}  [  2] _maybe_weak
; BARFOO-NEXT: 0x{{[0-9A-F]+}}  0x{{[0-9A-F]+}}  [  3] _foo

; LTO:      # Path: {{.*}}{{/|\\}}map-file.ll.tmp/foobar
; LTO-NEXT: # Arch: x86_64
; LTO-NEXT: # Object files:
; LTO-NEXT: [  0] linker synthesized
; LTO-NEXT: [  1] {{.*}}{{/|\\}}usr/lib{{/|\\}}libSystem.tbd
; LTO-NEXT: [  2] {{.*}}{{/|\\}}map-file.ll.tmp/foo.o
; LTO-NEXT: [  3] {{.*}}{{/|\\}}map-file.ll.tmp/bar.o
; LTO-NEXT: # Sections:
; LTO:      # Symbols:
; LTO-NEXT: # Address        Size             File   Name
; LTO-NEXT: 0x{{[0-9A-F]+}}  0x{{[0-9A-F]+}}  [  2]  _foo
; LTO-NEXT: 0x{{[0-9A-F]+}}  0x{{[0-9A-F]+}}  [  3]  _bar
; LTO-NEXT: 0x{{[0-9A-F]+}}  0x{{[0-9A-F]+}}  [  3]  _maybe_weak

;--- foo.ll
target triple = "x86_64-apple-darwin"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @foo() {
  ret void
}

;; This is weak in foo.ll but non-weak in bar.ll, so the definition in bar.ll
;; will always prevail. Check that the map file maps `maybe_weak` to bar.ll
;; regardless of the object file input order.
define weak_odr void @maybe_weak() {
  ret void
}

;--- bar.ll
target triple = "x86_64-apple-darwin"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @bar() {
  ret void
}

define void @maybe_weak() {
  ret void
}
