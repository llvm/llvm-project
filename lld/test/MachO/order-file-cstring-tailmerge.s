; REQUIRES: aarch64
; RUN: rm -rf %t && split-file %s %t

; RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/a.s -o %t/a.o
; RUN: %lld -dylib -arch arm64 --no-tail-merge-strings -order_file %t/orderfile.txt %t/a.o -o - | llvm-nm --numeric-sort --format=just-symbols - | FileCheck %s
; RUN: %lld -dylib -arch arm64 --tail-merge-strings -order_file %t/orderfile.txt %t/a.o -o - | llvm-nm --numeric-sort --format=just-symbols - | FileCheck %s --check-prefix=MERGED

; CHECK: _str2
; CHECK: _str1
; CHECK: _superstr2
; CHECK: _superstr3
; CHECK: _superstr1
; CHECK: _str3

; str1 has a higher priority than superstr1, so str1 must be ordered before
; str3, even though superstr1 is before superstr3 in the orderfile.

; MERGED: _superstr2
; MERGED: _str2
; MERGED: _superstr1
; MERGED: _str1
; MERGED: _superstr3
; MERGED: _str3

;--- a.s
.cstring
  _superstr1:
.asciz "superstr1"
  _str1:
.asciz "str1"
  _superstr2:
.asciz "superstr2"
  _str2:
.asciz "str2"
  _superstr3:
.asciz "superstr3"
  _str3:
.asciz "str3"

; TODO: We could use update_test_body.py to generate the hashes for the
; orderfile. Unfortunately, it seems that LLVM has a different hash
; implementation than the xxh64sum tool. See
; DeduplicatedCStringSection::getStringOffset() for hash details.
;
; while IFS="" read -r line; do
;     echo -n $line | xxh64sum | awk '{printf "CSTR;%010d", and(strtonum("0x"$1), 0x7FFFFFFF)}'
;     echo " # $line"
; done < orderfile.txt.template

;--- orderfile.txt
CSTR;1236462241 # str2
CSTR;1526669509 # str1
CSTR;1563550684 # superstr2
CSTR;1044337806 # superstr3
CSTR;262417687  # superstr1
CSTR;717161398  # str3
