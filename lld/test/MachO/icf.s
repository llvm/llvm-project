# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 %t/main.s -o %t/main.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 %t/abs.s -o %t/abs.o
# RUN: %lld -lSystem --icf=all -o %t/main %t/main.o %t/abs.o
# RUN: llvm-objdump -d --syms %t/main | FileCheck %s

# CHECK-LABEL: SYMBOL TABLE:
# CHECK:       [[#%x,MAIN:]] g   F __TEXT,__text _main
# CHECK:       [[#%x,A:]]    g   F __TEXT,__text _a1
# CHECK:       [[#%x,H:]]    g   F __TEXT,__text _h
# CHECK:       [[#%x,A]]     g   F __TEXT,__text _a2
# CHECK:       [[#%x,A]]     g   F __TEXT,__text _a3
# CHECK:       [[#%x,B:]]    g   F __TEXT,__text _b
# CHECK:       [[#%x,C:]]    g   F __TEXT,__text _c
# CHECK:       [[#%x,D:]]    g   F __TEXT,__text _d
# CHECK:       [[#%x,E:]]    g   F __TEXT,__text _e
# CHECK:       [[#%x,F:]]    g   F __TEXT,__text _f
# CHECK:       [[#%x,G:]]    g   F __TEXT,__text _g
# CHECK:       [[#%x,I:]]    g   F __TEXT,__text _i
# CHECK:       [[#%x,J:]]    g   F __TEXT,__text _j
# CHECK:       [[#%x,SR:]]   g   F __TEXT,__text _sr1
# CHECK:       [[#%x,SR]]    g   F __TEXT,__text _sr2
# CHECK:       [[#%x,MR:]]   g   F __TEXT,__text _mr1
# CHECK:       [[#%x,MR]]    g   F __TEXT,__text _mr2
### FIXME: Mutually-recursive functions with identical bodies (see below)
# COM:         [[#%x,XR:]]   g   F __TEXT,__text _xr1
# COM:         [[#%x,XR]]    g   F __TEXT,__text _xr2

# CHECK-LABEL: Disassembly of section __TEXT,__text:
# CHECK:       [[#%x,MAIN]] <_main>:
# CHECK-NEXT:  callq 0x[[#%x,A]]  <_a3>
# CHECK-NEXT:  callq 0x[[#%x,A]]  <_a3>
# CHECK-NEXT:  callq 0x[[#%x,A]]  <_a3>
# CHECK-NEXT:  callq 0x[[#%x,B]]  <_b>
# CHECK-NEXT:  callq 0x[[#%x,C]]  <_c>
# CHECK-NEXT:  callq 0x[[#%x,D]]  <_d>
# CHECK-NEXT:  callq 0x[[#%x,E]]  <_e>
# CHECK-NEXT:  callq 0x[[#%x,F]]  <_f>
# CHECK-NEXT:  callq 0x[[#%x,G]]  <_g>
# CHECK-NEXT:  callq 0x[[#%x,H]]  <_h>
# CHECK-NEXT:  callq 0x[[#%x,I]]  <_i>
# CHECK-NEXT:  callq 0x[[#%x,J]]  <_j>
# CHECK-NEXT:  callq 0x[[#%x,SR]] <_sr2>
# CHECK-NEXT:  callq 0x[[#%x,SR]] <_sr2>
# CHECK-NEXT:  callq 0x[[#%x,MR]] <_mr2>
# CHECK-NEXT:  callq 0x[[#%x,MR]] <_mr2>
### FIXME: Mutually-recursive functions with identical bodies (see below)
# COM-NEXT:    callq 0x[[#%x,XR]] <_xr2>
# COM-NEXT:    callq 0x[[#%x,XR]] <_xr2>

### TODO:
### * Fold: funcs only differ in alignment
### * No fold: func is weak? preemptable?

#--- abs.s
.subsections_via_symbols

.globl _abs1a, _abs1b, _abs2
_abs1a = 0xfeedfac3
_abs1b = 0xfeedfac3
_abs2 =  0xfeedf00d

#--- main.s
.subsections_via_symbols
.text
.globl _h
.alt_entry _h

### Fold: _a1 & _a2 have identical bodies, flags, relocs

.globl _a1
.p2align 2
_a1:
  callq _d
### No fold: _h is an alt entry past _a1
_h:
  mov ___nan@GOTPCREL(%rip), %rax
  callq ___isnan
  movabs $_abs1a, %rdx
  movl $0, %eax
  ret

.globl _a2
.p2align 2
_a2:
  callq _d
  mov ___nan@GOTPCREL(%rip), %rax
  callq ___isnan
  movabs $_abs1a, %rdx
  movl $0, %eax
  ret

### Fold: reference to absolute symbol with different name but identical value

.globl _a3
.p2align 2
_a3:
  callq _d
  mov ___nan@GOTPCREL(%rip), %rax
  callq ___isnan
  movabs $_abs1b, %rdx
  movl $0, %eax
  ret

### No fold: the absolute symbol value differs

.globl _b
.p2align 2
_b:
  callq _d
  mov ___nan@GOTPCREL(%rip), %rax
  callq ___isnan
  movabs $_abs2, %rdx
  movl $0, %eax
  ret

### No fold: _c has slightly different body from _a1 & _a2

.globl _c
.p2align 2
_c:
  callq _d
  mov ___nan@GOTPCREL(%rip), %rax
  callq ___isnan
  movabs $_abs1a, %rdx
  movl $1, %eax
  ret

### No fold: _d has the same body as _a1 & _a2, but _d is recursive!

.globl _d
.p2align 2
_d:
  callq _d
  mov ___nan@GOTPCREL(%rip), %rax
  callq ___isnan
  movabs $_abs1a, %rdx
  movl $0, %eax
  ret

### No fold: the function body is longer

.globl _e
.p2align 2
_e:
  callq _d
  mov ___nan@GOTPCREL(%rip), %rax
  callq ___isnan
  movabs $_abs1a, %rdx
  movl $0, %eax
  ret
  nop

### No fold: GOT referent dylib symbol differs

.globl _f
.p2align 2
_f:
  callq _d
  mov ___inf@GOTPCREL(%rip), %rax
  callq ___isnan
  movabs $_abs1a, %rdx
  movl $0, %eax
  ret

### No fold: call referent dylib symbol differs

.globl _g
.p2align 2
_g:
  callq _d
  mov ___nan@GOTPCREL(%rip), %rax
  callq ___isinf
  movabs $_abs1a, %rdx
  movl $0, %eax
  ret

### No fold: functions have personality and/or LSDA
### Mere presence of personality and/or LSDA isolates a function into its own
### equivalence class. We don't care if two functions happen to have identical
### personality & LSDA.

.globl _i
.p2align 2
_i:
  .cfi_startproc
  .cfi_personality 155, _my_personality0
  .cfi_lsda 16, _exception0
  .cfi_def_cfa_offset 16
  callq _d
  mov ___nan@GOTPCREL(%rip), %rax
  callq ___isnan
  movabs $_abs1a, %rdx
  movl $0, %eax
  ret
  .cfi_endproc

.globl _j
.p2align 2
_j:
  .cfi_startproc
  .cfi_personality 155, _my_personality0
  .cfi_lsda 16, _exception0
  .cfi_def_cfa_offset 16
  callq _d
  mov ___nan@GOTPCREL(%rip), %rax
  callq ___isnan
  movabs $_abs1a, %rdx
  movl $0, %eax
  ret
  .cfi_endproc

### Fold: Simple recursion

.globl _sr1
.p2align 2
_sr1:
  callq _sr1
  movl $0, %eax
  ret

.globl _sr2
.p2align 2
_sr2:
  callq _sr2
  movl $0, %eax
  ret

### Fold: Mutually-recursive functions with symmetric bodies

.globl _mr1
.p2align 2
_mr1:
  callq _mr1 # call myself
  callq _mr2 # call my twin
  movl $0, %eax
  ret

.globl _mr2
.p2align 2
_mr2:
  callq _mr2 # call myself
  callq _mr1 # call my twin
  movl $0, %eax
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
.p2align 2
_xr1:
  callq _xr1 # call myself
  callq _xr2 # call my twin
  movl $3, %eax
  ret

.globl _xr2
.p2align 2
_xr2:
  callq _xr1 # call my twin
  callq _xr2 # call myself
  movl $3, %eax
  ret

###

.globl _main
.p2align 2
_main:
  callq _a1
  callq _a2
  callq _a3
  callq _b
  callq _c
  callq _d
  callq _e
  callq _f
  callq _g
  callq _h
  callq _i
  callq _j
  callq _sr1
  callq _sr2
  callq _mr1
  callq _mr2
  callq _xr1
  callq _xr2
  ret

.globl _my_personality0
.p2align 2
_my_personality0:
  movl $0, %eax
  ret

.section __TEXT,__gcc_except_tab
.globl _exception0
_exception0:
  .space 1
