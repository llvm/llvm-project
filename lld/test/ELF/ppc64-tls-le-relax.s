# REQUIRES: ppc
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=powerpc64le %t/initexec -o %t/initexec.o
# RUN: llvm-mc -filetype=obj -triple=powerpc64le %t/defs -o %t/defs.o
# RUN: ld.lld %t/initexec.o %t/defs.o -o %t/out
# RUN: llvm-objdump -d --no-show-raw-insn %t/out | FileCheck %s

# CHECK-LABEL: <GetAddrT>:
# CHECK:         mflr 0
# CHECK-NEXT:    std 30, -16(1)
# CHECK-NEXT:    std 0, 16(1)
# CHECK-NEXT:    stdu 1, -48(1)
# CHECK-NEXT:    paddi 3, 13, -28672, 0
# CHECK-NEXT:    mr 30, 3
# CHECK-NEXT:    mr 3, 30
# CHECK-NEXT:    bl
# CHECK-NEXT:    mr 4, 30
# CHECK-NEXT:    addi 1, 1, 48
# CHECK-NEXT:    ld 0, 16(1)
# CHECK-NEXT:    ld 30, -16(1)
# CHECK-NEXT:    mtlr 0
# CHECK-NEXT:    b

## Generated From:
## extern __thread unsigned TGlobal;
## unsigned getConst(unsigned*);
## unsigned addVal(unsigned, unsigned*);
##
## unsigned GetAddrT() {
##   return addVal(getConst(&TGlobal), &TGlobal);
## }

//--- initexec
GetAddrT:
  mflr 0
  std 30, -16(1)
  std 0, 16(1)
  stdu 1, -48(1)
  pld 3, TGlobal@got@tprel@pcrel(0), 1
  add 30, 3, TGlobal@tls@pcrel
  mr      3, 30
  bl getConst@notoc
  mr      4, 30
  addi 1, 1, 48
  ld 0, 16(1)
  ld 30, -16(1)
  mtlr 0
  b addVal@notoc

## Generated From:
## __thread unsigned TGlobal;
##
## unsigned getConst(unsigned* A) {
##   return *A + 3;
## }
##
## unsigned addVal(unsigned A, unsigned* B) {
##   return A + *B;
## }

//--- defs
.globl  getConst
getConst:
  lwz 3, 0(3)
  addi 3, 3, 3
  clrldi  3, 3, 32
  blr

.globl  addVal
addVal:
  lwz 4, 0(4)
  add 3, 4, 3
  clrldi  3, 3, 32
  blr

.section        .tbss,"awT",@nobits
.globl  TGlobal
.p2align        2
TGlobal:
  .long   0
  .size   TGlobal, 4
