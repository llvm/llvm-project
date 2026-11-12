# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o
# RUN: ld.lld -shared --version-script=b.ver b.o -o b.so
# RUN: ld.lld --version-script=a.ver a.o b.so -o a0
# RUN: llvm-nm -D a0 | FileCheck %s
# RUN: ld.lld --version-script=a.ver b.so a.o -o a1
# RUN: llvm-nm -D a1 | FileCheck %s

# CHECK:      T a1{{$}}
# CHECK-NEXT: T a2{{$}}
# CHECK-NEXT: T b1{{$}}
# CHECK-NEXT: U b2{{$}}
# CHECK-NEXT: T c1@v1
# CHECK-NEXT: T c2@@v2
# CHECK-NEXT: T c3{{$}}
# CHECK-NEXT: T c4@@v3
# CHECK-NOT:  {{.}}

#--- a.s
.globl _start, a1, a2, a3, b1, c1, c2, c3, c4
_start:
a1: a2: ## defined in b.so and a
.hidden a3
a3: ## defined in b.so; hidden in a
b1: ## protected in b.so; defined in a
.symver c1, c1@v1, remove
c1: ## non-default version in b.so and a
c2: ## default version in b.so and a
c3: ## default version in b.so; unversioned in a
c4: ## default version in b.so; another version in a

.data
  .quad b2

#--- a.ver
v1 {};
v2 { c2; };
v3 { c4; };

#--- b.s
.globl a1, a2, a3, b1, b2, c1, c2, c3, c4
.type a1,@function
a1: a2: a3:
.protected b1, b2
b1:
b2:
.symver c1, c1@v1
c1:
c2:
c3:
c4:

#--- b.ver
v1 {};
v2 { c2; c3; c4; };
