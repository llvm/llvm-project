; REQUIRES: x86, xar
; RUN: rm -rf %t; split-file %s %t
; RUN: opt -module-summary %t/test.ll -o %t/test.o
; RUN: opt -module-summary %t/foo.ll -o %t/foo.o
; RUN: %lld -lSystem -bitcode_bundle %t/test.o %t/foo.o -o %t/test
; RUN: llvm-objdump --macho --section=__LLVM,__bundle %t/test | FileCheck %s
; RUN: llvm-readobj --macho-segment %t/test | FileCheck %s --check-prefix=SEGMENT

; CHECK:      Contents of (__LLVM,__bundle) section
; CHECK-NEXT: For (__LLVM,__bundle) section: xar header
; CHECK-NEXT:                   magic XAR_HEADER_MAGIC
; CHECK-NEXT:                    size 28
; CHECK-NEXT:                 version 1
; CHECK-NEXT:   toc_length_compressed
; CHECK-NEXT: toc_length_uncompressed
; CHECK-NEXT:               cksum_alg XAR_CKSUM_SHA1
; CHECK-NEXT: For (__LLVM,__bundle) section: xar table of contents:
; CHECK-NEXT: <?xml version="1.0" encoding="UTF-8"?>
; CHECK-NEXT: <xar>
; CHECK-NEXT:  <toc>
; CHECK-NEXT:   <checksum style="sha1">
; CHECK-NEXT:    <size>20</size>
; CHECK-NEXT:    <offset>0</offset>
; CHECK-NEXT:   </checksum>
; CHECK-NEXT:   <creation-time>{{.*}}</creation-time>
; CHECK-NEXT:  </toc>
; CHECK-NEXT: </xar>

;; __LLVM must directly precede __LINKEDIT.
; SEGMENT:        Name: __LLVM
; SEGMENT-NEXT:   Size: 152
; SEGMENT-NEXT:   vmaddr: 0x[[#%X,LLVM_ADDR:]]
; SEGMENT-NEXT:   vmsize: 0x[[#%X,LLVM_VMSIZE:]]
; SEGMENT-NEXT:   fileoff: [[#LLVM_OFF:]]
; SEGMENT-NEXT:   filesize: [[#LLVM_FILESIZE:]]
; SEGMENT-NEXT:   maxprot: rw-
; SEGMENT-NEXT:   initprot: rw-
; SEGMENT-NEXT:   nsects: 1
; SEGMENT-NEXT:   flags: 0x0
; SEGMENT-NEXT: }
; SEGMENT-NEXT: Segment {
; SEGMENT-NEXT:   Cmd: LC_SEGMENT_64
; SEGMENT-NEXT:   Name: __LINKEDIT
; SEGMENT-NEXT:   Size: 72
; SEGMENT-NEXT:   vmaddr: 0x[[#LLVM_ADDR + LLVM_VMSIZE]]
; SEGMENT-NEXT:   vmsize:
; SEGMENT-NEXT:   fileoff: [[#LLVM_OFF + LLVM_FILESIZE]]
; SEGMENT-NEXT:   filesize:
; SEGMENT-NEXT:   maxprot: r--
; SEGMENT-NEXT:   initprot: r--
; SEGMENT-NEXT:   nsects: 0
; SEGMENT-NEXT:   flags: 0x0
; SEGMENT-NEXT: }

;--- foo.ll
target triple = "x86_64-apple-darwin"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @foo() {
  ret void
}

;--- test.ll
target triple = "x86_64-apple-darwin"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @main() {
  ret void
}
