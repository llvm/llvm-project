# REQUIRES: x86
# RUN: rm -rf %t; mkdir %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/main.o
# RUN: %lld -lSystem -icf all -o %t/main %t/main.o
# RUN: llvm-objdump -d --syms %t/main | FileCheck %s

# CHECK-LABEL: SYMBOL TABLE:
# CHECK:       [[#%x,MAIN:]] g   F __TEXT,__text _main
# CHECK:       [[#%x,A:]]    g   F __TEXT,__text _a1
# CHECK:       [[#%x,A]]     g   F __TEXT,__text _a2
# CHECK:       [[#%x,C:]]    g   F __TEXT,__text _c
# CHECK:       [[#%x,D:]]    g   F __TEXT,__text _d
# CHECK:       [[#%x,E:]]    g   F __TEXT,__text _e
# CHECK:       [[#%x,F:]]    g   F __TEXT,__text _f
# CHECK:       [[#%x,G:]]    g   F __TEXT,__text _g
# CHECK:       [[#%x,SR:]]   g   F __TEXT,__text _sr1
# CHECK:       [[#%x,SR]]    g   F __TEXT,__text _sr2
# CHECK:       [[#%x,MR:]]   g   F __TEXT,__text _mr1
# CHECK:       [[#%x,MR]]    g   F __TEXT,__text _mr2
### FIXME: Mutually-recursive functions with identical bodies (see below)
# COM:         [[#%x,XR:]]   g   F __TEXT,__text _xr1
# COM:         [[#%x,XR]]    g   F __TEXT,__text _xr2

# CHECK-LABEL: Disassembly of section __TEXT,__text:
# CHECK:       [[#%x,MAIN]] <_main>:
# CHECK-NEXT:  callq 0x[[#%x,A]]  <_a2>
# CHECK-NEXT:  callq 0x[[#%x,A]]  <_a2>
# CHECK-NEXT:  callq 0x[[#%x,C]]  <_c>
# CHECK-NEXT:  callq 0x[[#%x,D]]  <_d>
# CHECK-NEXT:  callq 0x[[#%x,E]]  <_e>
# CHECK-NEXT:  callq 0x[[#%x,F]]  <_f>
# CHECK-NEXT:  callq 0x[[#%x,G]]  <_g>
# CHECK-NEXT:  callq 0x[[#%x,SR]] <_sr2>
# CHECK-NEXT:  callq 0x[[#%x,SR]] <_sr2>
# CHECK-NEXT:  callq 0x[[#%x,MR]] <_mr2>
# CHECK-NEXT:  callq 0x[[#%x,MR]] <_mr2>
### FIXME: Mutually-recursive functions with identical bodies (see below)
# COM-NEXT:    callq 0x[[#%x,XR]] <_xr2>
# COM-NEXT:    callq 0x[[#%x,XR]] <_xr2>

### TODO:
### * Fold: funcs only differ in alignment
### * No fold: func has personality/LSDA
### * No fold: reloc references to absolute symbols with different values
### * No fold: func is weak? preemptable?
### * No fold: relocs to N_ALT_ENTRY symbols

.subsections_via_symbols
.text

### Fold: _a1 & _a2 have identical bodies, flags, relocs

.globl _a1
.p2align 4, 0x90
_a1:
  callq _d
  mov ___nan@GOTPCREL(%rip), %rax
  callq ___isnan
  movl $0, %eax
  ret

.globl _a2
.p2align 4, 0x90
_a2:
  callq _d
  mov ___nan@GOTPCREL(%rip), %rax
  callq ___isnan
  movl $0, %eax
  ret

### No fold: _c has slightly different body from _a1 & _a2

.globl _c
.p2align 4, 0x90
_c:
  callq _d
  mov ___nan@GOTPCREL(%rip), %rax
  callq ___isnan
  movl $1, %eax
  ret

### No fold: _d has the same body as _a1 & _a2, but _d is recursive!

.globl _d
.p2align 4, 0x90
_d:
  callq _d
  mov ___nan@GOTPCREL(%rip), %rax
  callq ___isnan
  movl $0, %eax
  ret

### No fold: the body of _e is longer

.globl _e
.p2align 4, 0x90
_e:
  callq _d
  mov ___nan@GOTPCREL(%rip), %rax
  callq ___isnan
  movl $0, %eax
  ret
  nop

### No fold: the dylib symbols differ

.globl _f
.p2align 4, 0x90
_f:
  callq _d
  mov ___inf@GOTPCREL(%rip), %rax
  callq ___isnan
  movl $0, %eax
  ret

.globl _g
.p2align 4, 0x90
_g:
  callq _d
  mov ___inf@GOTPCREL(%rip), %rax
  callq ___isinf
  movl $0, %eax
  ret

### Fold: Simple recursion

.globl _sr1
.p2align 4, 0x90
_sr1:
  callq _sr1
  movl $2, %eax
  ret

.globl _sr2
.p2align 4, 0x90
_sr2:
  callq _sr2
  movl $2, %eax
  ret

### Fold: Mutually-recursive functions with symmetric bodies

.globl _mr1
.p2align 4, 0x90
_mr1:
  callq _mr1 # call myself
  callq _mr2 # call my twin
  movl $1, %eax
  ret

.globl _mr2
.p2align 4, 0x90
_mr2:
  callq _mr2 # call myself
  callq _mr1 # call my twin
  movl $1, %eax
  ret

### Fold: Mutually-recursive functions with identical bodies
###
### FIXME: This test is currently broken. Recursive call sites have no relocs
### and the non-zero displacement field is already written to the section
### data, while non-recursive call sites use symbol relocs and section data
### contains zeros in the displacement field. Thus, ICF's equalsConstant()
### finds that the section data doesn't match.
###
### ELF folds this case properly because it emits symbol relocs for all calls,
### even recursive ones.

.globl _xr1
.p2align 4, 0x90
_xr1:
  callq _xr1 # call myself
  callq _xr2 # call my twin
  movl $3, %eax
  ret

.globl _xr2
.p2align 4, 0x90
_xr2:
  callq _xr1 # call my twin
  callq _xr2 # call myself
  movl $3, %eax
  ret

###

.globl _main
.p2align 4, 0x90
_main:
  callq _a1
  callq _a2
  callq _c
  callq _d
  callq _e
  callq _f
  callq _g
  callq _sr1
  callq _sr2
  callq _mr1
  callq _mr2
  callq _xr1
  callq _xr2
  ret
