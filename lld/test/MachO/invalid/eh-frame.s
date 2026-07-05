# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos10.15 %t/too-small-1.s -o %t/too-small-1.o
# RUN: not %lld -lSystem -dylib %t/too-small-1.o -o /dev/null 2>&1 | FileCheck %s --check-prefix TOO-SMALL-1
# TOO-SMALL-1: error: {{.*}}too-small-1.o:(__eh_frame+0x0): CIE/FDE too small

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos10.15 %t/too-small-2.s -o %t/too-small-2.o
# RUN: not %lld -lSystem -dylib %t/too-small-2.o -o /dev/null 2>&1 | FileCheck %s --check-prefix TOO-SMALL-2
# TOO-SMALL-2: error: {{.*}}too-small-2.o:(__eh_frame+0x0): CIE/FDE extends past the end of the section

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos10.15 %t/personality-enc.s -o %t/personality-enc.o
# RUN: not %lld -lSystem -dylib %t/personality-enc.o -o /dev/null 2>&1 | FileCheck %s --check-prefix PERS-ENC
# PERS-ENC: error: {{.*}}personality-enc.o:(__eh_frame+0x12): unexpected personality encoding 0xb

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos10.15 %t/pointer-enc.s -o %t/pointer-enc.o
# RUN: not %lld -lSystem -dylib %t/pointer-enc.o -o /dev/null 2>&1 | FileCheck %s --check-prefix PTR-ENC
# PTR-ENC: error: {{.*}}pointer-enc.o:(__eh_frame+0x11): unexpected pointer encoding 0x12

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos10.15 %t/string-err.s -o %t/string-err.o
# RUN: not %lld -lSystem -dylib %t/string-err.o -o /dev/null 2>&1 | FileCheck %s --check-prefix STR
# STR: error: {{.*}}string-err.o:(__eh_frame+0x9): corrupted CIE (failed to read string)

#--- too-small-1.s
.p2align 3
.section __TEXT,__eh_frame
.short 0x3

.subsections_via_symbols

#--- too-small-2.s
.p2align 3
.section __TEXT,__eh_frame
.long 0x3  # length

.subsections_via_symbols

#--- personality-enc.s
.p2align 3
.section __TEXT,__eh_frame

.long 0x14   # length
.long 0      # CIE offset
.byte 1      # version
.asciz "zPR" # aug string
.byte 0x01   # code alignment
.byte 0x78   # data alignment
.byte 0x10   # return address register
.byte 0x01   # aug length
.byte 0x0b   # personality encoding
.long 0xffff # personality pointer
.byte 0x10   # pointer encoding
.space 1     # pad to alignment

.subsections_via_symbols

#--- pointer-enc.s
.p2align 3
.section __TEXT,__eh_frame

.long 0x14  # length
.long 0     # CIE offset
.byte 1     # version
.asciz "zR" # aug string
.byte 0x01  # code alignment
.byte 0x78  # data alignment
.byte 0x10  # return address register
.byte 0x01  # aug length
.byte 0x12  # pointer encoding
.space 7    # pad to alignment

.subsections_via_symbols

#--- string-err.s
.p2align 3
.section __TEXT,__eh_frame

.long 0x7   # length
.long 0     # CIE offset
.byte 1     # version
.ascii "zR" # invalid aug string

.subsections_via_symbols
