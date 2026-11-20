# REQUIRES: x86

# Verify that metadata/resource children and string-tail-merge candidates obey
# late LARGEST replacement just like ordinary code and data sections.

# RUN: split-file %s %t.dir
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/meta-small.s -o %t.meta-small.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/meta-large.s -o %t.meta-large.obj
# RUN: lld-link /dll /noentry /nodefaultlib /include:leader /guard:cf \
# RUN:   /guard:ehcont %t.meta-small.obj %t.meta-large.obj /out:%t.meta.dll \
# RUN:   2>&1 | FileCheck %s --allow-empty --check-prefix=NO-ERROR
# RUN: llvm-objdump -s %t.meta.dll | FileCheck %s --check-prefix=META \
# RUN:   --implicit-check-not=11111111 --implicit-check-not=aaaaaaaa \
# RUN:   --implicit-check-not=534d414c

# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/string-small.s -o %t.string-small.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/string-large.s -o %t.string-large.obj
# RUN: lld-link /dll /noentry /nodefaultlib '/include:??_C@largest' \
# RUN:   %t.string-small.obj %t.string-large.obj /out:%t.string.dll
# RUN: llvm-objdump -s %t.string.dll | FileCheck %s --check-prefix=STRING \
# RUN:   --implicit-check-not=534d414c

# MinGW implicitly associates .pdata$*, .xdata$* and .eh_frame$* COMDATs with
# the matching function section. Those children must follow a late LARGEST
# replacement as well.
# RUN: llvm-mc -triple x86_64-windows-gnu -filetype=obj \
# RUN:   %t.dir/mingw-small.s -o %t.mingw-small.obj
# RUN: llvm-mc -triple x86_64-windows-gnu -filetype=obj \
# RUN:   %t.dir/mingw-large.s -o %t.mingw-large.obj
# RUN: lld-link /lldmingw /dll /noentry /nodefaultlib /include:leader \
# RUN:   %t.mingw-small.obj %t.mingw-large.obj /out:%t.mingw.dll
# RUN: llvm-objdump -s %t.mingw.dll | FileCheck %s --check-prefix=MINGW \
# RUN:   --implicit-check-not=11111111 --implicit-check-not=aaaaaaaa \
# RUN:   --implicit-check-not=534d414c

# NO-ERROR-NOT: error:
# META: Contents of section .text:
# META: 44444444
# META: Contents of section .rdata:
# META: 55555555
# META-NOT: Contents of section .rsrc:
# STRING: Contents of section .rdata:
# STRING: 4c415247 45522d53 5452494e 4700
# MINGW: 44444444
# MINGW: 4752414c

#--- meta-small.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .long 0x11111111
small_end:

        .section .gfids$y, "dr", associative, leader
        .symidx leader
        .section .giats$y, "dr", associative, leader
        .symidx leader
        .section .gljmp$y, "dr", associative, leader
        .symidx leader
        .section .gehcont$y, "dr", associative, leader
        .symidx leader

        .section .xdata, "r", associative, leader
small_unwind:
        .long 0xaaaaaaaa
        .section .pdata, "r", associative, leader
        .rva leader, small_end, small_unwind
        .section .eh_frame, "r", associative, leader
        .long 0x534d414c
        .section .rsrc$01, "dr", associative, leader
        .long 0x534d414c

#--- meta-large.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .long 0x44444444
        .space 28, 0x44
large_end:

        .section .gfids$y, "dr", associative, leader
        .symidx leader
        .section .giats$y, "dr", associative, leader
        .symidx leader
        .section .gljmp$y, "dr", associative, leader
        .symidx leader
        .section .gehcont$y, "dr", associative, leader
        .symidx leader

        .section .xdata, "r", associative, leader
large_unwind:
        .long 0x55555555
        .section .pdata, "r", associative, leader
        .rva leader, large_end, large_unwind
        .section .eh_frame, "r", associative, leader
        .long 0x4c415247

#--- string-small.s
        .section .rdata, "dr", largest, "??_C@largest"
        .globl "??_C@largest"
"??_C@largest":
        .asciz "SMALL"

#--- string-large.s
        .section .rdata, "dr", largest, "??_C@largest"
        .globl "??_C@largest"
"??_C@largest":
        .asciz "LARGER-STRING"

#--- mingw-small.s
        .section .xdata$leader, "dr"
        .linkonce discard
        .long 0xaaaaaaaa
        .section .pdata$leader, "dr"
        .linkonce discard
        .long 0x534d414c
        .long 0x534d414c
        .long 0x534d414c
        .section .eh_frame$leader, "dr"
        .linkonce discard
        .long 0x534d414c
        .section .text$leader, "xr", largest, leader
        .globl leader
leader:
        .long 0x11111111

#--- mingw-large.s
        .section .xdata$leader, "dr"
        .linkonce discard
        .long 0x55555555
        .section .pdata$leader, "dr"
        .linkonce discard
        .long 0x4c415247
        .long 0x4c415247
        .long 0x4c415247
        .section .eh_frame$leader, "dr"
        .linkonce discard
        .long 0x4c415247
        .section .text$leader, "xr", largest, leader
        .globl leader
leader:
        .long 0x44444444
        .space 28, 0x44
