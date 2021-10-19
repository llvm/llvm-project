# REQUIRES: x86

# RUN: llvm-mc -triple=x86_64-windows-msvc -defsym drectve=0 %s -filetype=obj -o %t.obj
# RUN: echo "EXPORTS  ??_GFoo@@UEAAPEAXI@Z" > %t.def

# RUN: lld-link %t.obj -entry:dllmain -dll -export:'??_GFoo@@UEAAPEAXI@Z' -out:%t.1.dll 2>&1 | FileCheck %s

# RUN: lld-link %t.obj -entry:dllmain -dll -def:%t.def -out:%t.2.dll 2>&1 | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc -defsym drectve=1 %s -o %t.drectve.obj
# RUN: lld-link %t.drectve.obj -entry:dllmain -dll -out:%t.3.dll 2>&1 | FileCheck %s

# CHECK: export of deleting dtor:{{.*}}Foo{{.*}}

        .text
        .globl  dllmain
        .globl  "??_GFoo@@UEAAPEAXI@Z"
dllmain:
        ret
"??_GFoo@@UEAAPEAXI@Z":
        ret

.if drectve==1
        .section .drectve
        .ascii "-export:??_GFoo@@UEAAPEAXI@Z"
.endif
