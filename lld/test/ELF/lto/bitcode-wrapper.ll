; REQUIRES: x86

;; The LLVM bitcode format allows for an optional wrapper header. This test
;; shows that LLD can handle bitcode wrapped in this way, and also that an
;; invalid offset in the wrapper header is handled cleanly.

; RUN: rm -rf %t
; RUN: split-file %s %t
; RUN: llvm-as %t/ir.ll -o %t.bc

;; Basic case:
; RUN: %python %t/wrap_bitcode.py %t.bc %t.o 0 0x14
; RUN: ld.lld %t.o -o %t.elf
; RUN: llvm-readelf -s %t.elf | FileCheck %s

;; Padding between wrapper header and body:
; RUN: %python %t/wrap_bitcode.py %t.bc %t.o 0x10 0x24
; RUN: ld.lld %t.o -o %t.elf
; RUN: llvm-readelf -s %t.elf | FileCheck %s

; CHECK: _start

;; Invalid offset past end of file:
; RUN: %python %t/wrap_bitcode.py %t.bc %t2.o 0x10 0xffffffff
; RUN: not ld.lld %t2.o -o %t2.elf 2>&1 | FileCheck %s --check-prefix=ERR1 -DFILE=%t2.o

; ERR1: error: [[FILE]]: Invalid bitcode wrapper header

;; Invalid offset within file:
; RUN: %python %t/wrap_bitcode.py %t.bc %t3.o 0x10 0x14
; RUN: not ld.lld %t3.o -o %t3.elf 2>&1 | FileCheck %s --check-prefix=ERR2 -DFILE=%t3.o

; ERR2: error: [[FILE]]: file doesn't start with bitcode header

;--- ir.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_start = global i32 0

;--- wrap_bitcode.py
## Arguments are: input file, output file, padding size, offset value.
import struct
import sys

with open(sys.argv[1], 'rb') as input:
    bitcode = input.read()

padding = int(sys.argv[3], 16) * b'\0'
offset = int(sys.argv[4], 16)
header = struct.pack('<IIIII', 0x0B17C0DE, 0, offset, len(bitcode), 0)
with open(sys.argv[2], 'wb') as output:
    output.write(header)
    output.write(padding)
    output.write(bitcode)
