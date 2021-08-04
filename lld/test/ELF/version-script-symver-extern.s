# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

# RUN: echo 'v1 { local: extern "C++" { "foo1()"; }; }; v2 { local: extern "C++" { "foo2()"; }; };' > %t1.script
# RUN: ld.lld --version-script %t1.script -shared %t.o -o %t1.so
# RUN: llvm-readelf --dyn-syms %t1.so | FileCheck --check-prefix=EXACT %s
# EXACT:      UND
# EXACT-NEXT: [[#]] _start{{$}}
# EXACT-NEXT: [[#]] _Z4foo3i@v1
# EXACT-NEXT: [[#]] _Z4foo4i@@v2
# EXACT-NOT:  {{.}}

# RUN: echo 'v1 { global: *; local: extern "C++" {foo*;}; }; v2 {};' > %t2.script
# RUN: ld.lld --version-script %t2.script -shared %t.o -o %t2.so
# RUN: llvm-readelf --dyn-syms %t2.so | FileCheck --check-prefix=MIX1 %s
# MIX1:      UND
# MIX1-NEXT: [[#]] _start@@v1
# MIX1-NEXT: [[#]] _Z4foo3i@v1
# MIX1-NEXT: [[#]] _Z4foo4i@@v2
# MIX1-NOT:  {{.}}

# RUN: echo 'v1 { global: extern "C++" {foo*;}; local: *; }; v2 { global: extern "C++" {"foo4(int)";}; local: *; };' > %t3.script
# RUN: ld.lld --version-script %t3.script -shared %t.o -o %t3.so
# RUN: llvm-readelf --dyn-syms %t3.so | FileCheck --check-prefix=MIX2 %s
# MIX2:      UND
# MIX2-NEXT: [[#]] _Z4foo1v@@v1
# MIX2-NEXT: [[#]] _Z4foo2v@@v1
# MIX2-NEXT: [[#]] _Z4foo3i@v1
# MIX2-NEXT: [[#]] _Z4foo4i@@v2
# MIX2-NOT:  {{.}}

.globl _Z4foo1v; _Z4foo1v: ret
.globl _Z4foo2v; _Z4foo2v: ret
.globl _Z4foo3i; .symver _Z4foo3i,_Z4foo3i@v1,remove; _Z4foo3i: ret
.globl _Z4foo4i; .symver _Z4foo4i,_Z4foo4i@@@v2; _Z4foo4i: ret

.globl _start; _start: ret
