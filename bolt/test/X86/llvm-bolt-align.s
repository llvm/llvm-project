# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   --defsym VARIATION=0 %s -o %t.a.o
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   --defsym VARIATION=1 %s -o %t.b.o
# RUN: %clang %cflags %t.a.o -o %t.a.exe -Wl,-q
# RUN: %clang %cflags %t.b.o -o %t.b.exe -Wl,-q
# RUN: llvm-bolt-align %t.a.exe %t.b.exe -o-a=%t.a -o-b=%t.b 2>&1 | FileCheck %s
# RUN: llvm-nm -n --defined-only %t.a \
# RUN:   | awk '$3 ~ /^(main|funcb|funce)$/ { print $1, $3 }' > %t.a.nm
# RUN: llvm-nm -n --defined-only %t.b \
# RUN:   | awk '$3 ~ /^(main|funcb|funce)$/ { print $1, $3 }' > %t.b.nm
# RUN: diff -u %t.a.nm %t.b.nm

# CHECK: BOLT-ALIGN: pinned 3 functions (60%)
# CHECK: BOLT-ALIGN: wrote aligned binaries

  .macro emit_main
  .globl main
  .type main, %function
main:
  callq funca
  callq funcb
  callq funce
  retq
.size main, .-main
  .endm

  .macro emit_funca
  .globl funca
  .type funca, %function
funca:
.if VARIATION==1
  .rept 32
  nop
  .endr
.endif
  retq
.size funca, .-funca
  .endm

  .macro emit_funcb
  .globl funcb
  .type funcb, %function
funcb:
  retq
.size funcb, .-funcb
  .endm

  .macro emit_funcc
  .globl funcc
  .type funcc, %function
funcc:
  retq
.size funcc, .-funcc
  .endm

  .macro emit_funcd
  .globl funcd
  .type funcd, %function
funcd:
  retq
.size funcd, .-funcd
  .endm

  .macro emit_funce
  .globl funce
  .type funce, %function
funce:
  retq
.size funce, .-funce
  .endm

.if VARIATION==0
  emit_main
  emit_funca
  emit_funcb
  emit_funcc
  emit_funce
.else
  emit_main
  emit_funcb
  emit_funca
  emit_funcd
  emit_funce
.endif
