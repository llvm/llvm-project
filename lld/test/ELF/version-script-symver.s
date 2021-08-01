# REQUIRES: x86
## Test how .symver interacts with --version-script.
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

# RUN: echo 'v1 { local: foo1; }; v2 { local: foo2; };' > %t1.script
# RUN: ld.lld --version-script %t1.script -shared %t.o -o %t1.so
# RUN: llvm-readelf --dyn-syms %t1.so | FileCheck --check-prefix=EXACT %s
# EXACT:      UND
# EXACT-NEXT: [[#]] foo3{{$}}
# EXACT-NEXT: [[#]] foo4@@v2
# EXACT-NEXT: [[#]] _start{{$}}
# EXACT-NEXT: [[#]] foo3@v1
# EXACT-NOT:  {{.}}

# RUN: echo 'v1 { local: foo*; }; v2 {};' > %t2.script
# RUN: ld.lld --version-script %t2.script -shared %t.o -o %t2.so
# RUN: llvm-readelf --dyn-syms %t2.so | FileCheck --check-prefix=WC %s
# WC:      UND
# WC-NEXT: [[#]] foo4@@v2
# WC-NEXT: [[#]] _start{{$}}
# WC-NEXT: [[#]] foo3@v1
# WC-NOT:  {{.}}

# RUN: echo 'v1 { global: *; local: foo*; }; v2 {};' > %t3.script
# RUN: ld.lld --version-script %t3.script -shared %t.o -o %t3.so
# RUN: llvm-readelf --dyn-syms %t3.so | FileCheck --check-prefix=MIX1 %s
# MIX1:      UND
# MIX1-NEXT: [[#]] foo4@@v2
# MIX1-NEXT: [[#]] _start@@v1
# MIX1-NEXT: [[#]] foo3@v1
# MIX1-NOT:  {{.}}

# RUN: echo 'v1 { global: foo*; local: *; }; v2 { global: foo4; local: *; };' > %t4.script
# RUN: ld.lld --version-script %t4.script -shared %t.o -o %t4.so
# RUN: llvm-readelf --dyn-syms %t4.so | FileCheck --check-prefix=MIX2 %s
# MIX2:      UND
# MIX2-NEXT: [[#]] foo1@@v1
# MIX2-NEXT: [[#]] foo2@@v1
# MIX2-NEXT: [[#]] foo3@@v1
# MIX2-NEXT: [[#]] foo4@@v2
# MIX2-NEXT: [[#]] foo3@v1
# MIX2-NOT:  {{.}}

.globl foo1; foo1: ret
.globl foo2; foo2: ret
.globl foo3; .symver foo3,foo3@v1; foo3: ret
.globl foo4; .symver foo4,foo4@@v2; foo4: ret

.globl _start; _start: ret
