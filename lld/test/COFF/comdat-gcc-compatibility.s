# REQUIRES: x86
# RUN: llvm-mc %s -triple x86_64-pc-win32 -defsym obj=0 -filetype=obj -o %t1.obj
# RUN: llvm-mc %s -triple x86_64-pc-win32 -defsym obj=1 -filetype=obj -o %t2.obj
# RUN: lld-link /lldmingw /noentry /dll %t1.obj %t2.obj /out:%t3.dll
# RUN: not lld-link /noentry /dll %t1.obj %t2.obj /out:%t3.dll
.if obj==0
        .section .text$nm, "", discard, symbol
.else
        .section .text$nm, "", same_size, symbol
.endif
        .globl symbol
symbol:
        .long 1
