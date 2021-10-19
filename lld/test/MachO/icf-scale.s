# REQUIRES: x86
# RUN: rm -rf %t*

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: %lld -lSystem --icf=all -o %t %t.o
# RUN: llvm-objdump -d --syms %t | FileCheck %s

## When ICF has fewer than 1 Ki functions to segregate into equivalence classes,
## it uses a sequential algorithm to avoid the overhead of threading.
## At 1 Ki functions or more, when threading begins to pay-off, ICF employs its
## parallel segregation algorithm. Here we generate 4 Ki functions to exercise
## the parallel algorithm. There are 4 unique function bodies, each replicated
## 1 Ki times. The resulting folded program should retain one instance for each
## of the four unique functions.

# CHECK-LABEL: SYMBOL TABLE:
# CHECK: [[#%x,G0:]] g   F __TEXT,__text _g000000
# CHECK: [[#%x,G1:]] g   F __TEXT,__text _g100000
# CHECK: [[#%x,G2:]] g   F __TEXT,__text _g200000
# CHECK: [[#%x,G3:]] g   F __TEXT,__text _g300000
## . . . many intervening _gXXXXXX symbols
# CHECK: [[#%x,G0]]  g   F __TEXT,__text _g033333
# CHECK: [[#%x,G1]]  g   F __TEXT,__text _g133333
# CHECK: [[#%x,G2]]  g   F __TEXT,__text _g233333
# CHECK: [[#%x,G3]]  g   F __TEXT,__text _g333333

# CHECK-LABEL: Disassembly of section __TEXT,__text:
# CHECK-DAG: [[#%x,G0]]  <_g033333>:
# CHECK-DAG: [[#%x,G1]]  <_g133333>:
# CHECK-DAG: [[#%x,G2]]  <_g233333>:
# CHECK-DAG: [[#%x,G3]]  <_g333333>:
# CHECK-NOT: [[#]]       <_g{{.*}}>:

.subsections_via_symbols
.text
.p2align 2

.macro gen_4 c
  .globl _g0\c, _g1\c, _g2\c, _g3\c
  _g0\c:; movl $0, %eax; ret
  _g1\c:; movl $1, %eax; ret
  _g2\c:; movl $2, %eax; ret
  _g3\c:; movl $3, %eax; ret
.endm

.macro gen_16 c
  gen_4 0\c
  gen_4 1\c
  gen_4 2\c
  gen_4 3\c
.endm

.macro gen_64 c
  gen_16 0\c
  gen_16 1\c
  gen_16 2\c
  gen_16 3\c
.endm

.macro gen_256 c
  gen_64 0\c
  gen_64 1\c
  gen_64 2\c
  gen_64 3\c
.endm

.macro gen_1024 c
  gen_256 0\c
  gen_256 1\c
  gen_256 2\c
  gen_256 3\c
.endm

gen_1024 0
gen_1024 1
gen_1024 2
gen_1024 3

.globl _main
_main:
  ret
