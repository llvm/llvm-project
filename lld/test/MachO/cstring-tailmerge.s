; REQUIRES: aarch64
; RUN: rm -rf %t && split-file %s %t

; RUN: sed "s/<ALIGN>/0/g" %t/align.s.template > %t/align-1.s
; RUN: sed "s/<ALIGN>/1/g" %t/align.s.template > %t/align-2.s
; RUN: sed "s/<ALIGN>/2/g" %t/align.s.template > %t/align-4.s

; RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/first.s -o %t/first.o
; RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/align-1.s -o %t/align-1.o
; RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/align-2.s -o %t/align-2.o
; RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/align-4.s -o %t/align-4.o

; RUN: %lld -dylib -arch arm64 --tail-merge-strings %t/first.o %t/align-1.o -o %t/align-1
; RUN: llvm-objdump --macho --section="__TEXT,__cstring" --syms %t/align-1 | FileCheck %s --check-prefixes=CHECK,ALIGN1

; RUN: %lld -dylib -arch arm64 --tail-merge-strings %t/first.o %t/align-2.o -o %t/align-2
; RUN: llvm-objdump --macho --section="__TEXT,__cstring" --syms %t/align-2 | FileCheck %s --check-prefixes=CHECK,ALIGN2

; RUN: %lld -dylib -arch arm64 --tail-merge-strings %t/first.o %t/align-4.o -o %t/align-4
; RUN: llvm-objdump --macho --section="__TEXT,__cstring" --syms %t/align-4 | FileCheck %s --check-prefixes=CHECK,ALIGN4

; CHECK: Contents of (__TEXT,__cstring) section
; CHECK: [[#%.16x,START:]] get awkward offset{{$}}

; ALIGN1: [[#%.16x,START+19]] myotherlongstr{{$}}
; ALIGN1: [[#%.16x,START+19+15]] otherstr{{$}}

; ALIGN2: [[#%.16x,START+20]] myotherlongstr{{$}}
; ALIGN2: [[#%.16x,START+20+16]] longstr{{$}}
; ALIGN2: [[#%.16x,START+20+16+8]] otherstr{{$}}
; ALIGN2: [[#%.16x,START+20+16+8+10]] str{{$}}

; ALIGN4: [[#%.16x,START+20]] myotherlongstr{{$}}
; ALIGN4: [[#%.16x,START+20+16]] otherlongstr{{$}}
; ALIGN4: [[#%.16x,START+20+16+16]] longstr{{$}}
; ALIGN4: [[#%.16x,START+20+16+16+8]] otherstr{{$}}
; ALIGN4: [[#%.16x,START+20+16+16+8+12]] str{{$}}

; CHECK: SYMBOL TABLE:

; ALIGN1: [[#%.16x,START+19]] l     O __TEXT,__cstring _myotherlongstr
; ALIGN1: [[#%.16x,START+21]] l     O __TEXT,__cstring _otherlongstr
; ALIGN1: [[#%.16x,START+26]] l     O __TEXT,__cstring _longstr
; ALIGN1: [[#%.16x,START+34]] l     O __TEXT,__cstring _otherstr
; ALIGN1: [[#%.16x,START+39]] l     O __TEXT,__cstring _str

; ALIGN2: [[#%.16x,START+20]] l     O __TEXT,__cstring _myotherlongstr
; ALIGN2: [[#%.16x,START+20+2]] l     O __TEXT,__cstring _otherlongstr
; ALIGN2: [[#%.16x,START+20+16]] l     O __TEXT,__cstring _longstr
; ALIGN2: [[#%.16x,START+20+16+8]] l     O __TEXT,__cstring _otherstr
; ALIGN2: [[#%.16x,START+20+16+8+10]] l     O __TEXT,__cstring _str

; ALIGN4: [[#%.16x,START+20]] l     O __TEXT,__cstring _myotherlongstr
; ALIGN4: [[#%.16x,START+20+16]] l     O __TEXT,__cstring _otherlongstr
; ALIGN4: [[#%.16x,START+20+16+16]] l     O __TEXT,__cstring _longstr
; ALIGN4: [[#%.16x,START+20+16+16+8]] l     O __TEXT,__cstring _otherstr
; ALIGN4: [[#%.16x,START+20+16+16+8+12]] l     O __TEXT,__cstring _str

;--- first.s
.cstring
.p2align 2
.asciz "get awkward offset"  ; length = 19

;--- align.s.template
.cstring

.p2align <ALIGN>
  _myotherlongstr:
.asciz "myotherlongstr"      ; length = 15

.p2align <ALIGN>
  _otherlongstr:
.asciz   "otherlongstr"      ; length = 13, tail offset = 2

.p2align <ALIGN>
  _longstr:
.asciz        "longstr"      ; length = 8, tail offset = 7

.p2align <ALIGN>
  _otherstr:
.asciz       "otherstr"      ; length = 9

.p2align <ALIGN>
  _str:
.asciz            "str"      ; length = 4, tail offset = 5
