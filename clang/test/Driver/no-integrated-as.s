; RUN: %clang -### -no-integrated-as -c %s 2>&1 | FileCheck %s -check-prefix IAS
; Windows and PS5 don't support no-integrated-as
; XFAIL: target={{.*-windows-msvc}}, target={{.*-ps5}}
;
; Make sure the current file's filename appears in the output.
; We can't generically match on the assembler name, so we just make sure
; the filename is in the output.
; IAS: no-integrated-as.s
