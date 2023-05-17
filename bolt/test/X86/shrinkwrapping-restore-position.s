# This checks that shrink wrapping uses the red zone defined in the X86 ABI by
# placing restores that access elements already deallocated by the stack.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -nostdlib
# RUN: llvm-bolt -relocs %t.exe -o %t.out -data %t.fdata \
# RUN:     -frame-opt=all -simplify-conditional-tail-calls=false \
# RUN:     -experimental-shrink-wrapping \
# RUN:     -eliminate-unreachable=false | FileCheck %s
# RUN: llvm-objdump -d %t.out --print-imm-hex | \
# RUN:   FileCheck --check-prefix CHECK-OBJDUMP %s


# Here we create a CFG where the restore position matches the previous (deleted)
# restore position. Shrink wrapping then will put a stack access to an element
# that was deallocated at the previously deleted POP, which falls in the red
# zone and should be safe for X86 Linux ABI.
  .globl _start
  .type _start, %function
_start:
  .cfi_startproc
# FDATA: 0 [unknown] 0 1 _start 0 0 1
  push  %rbp
  mov   %rsp, %rbp
  push  %rbx
  push  %r14
  subq  $0x20, %rsp
b:  je  hot_path
# FDATA: 1 _start #b# 1 _start #hot_path# 0 1
cold_path:
  mov %r14, %rdi
  mov %rbx, %rdi
  movq rel(%rip), %rdi  # Add this to create a relocation and run bolt w/ relocs
  leaq -0x20(%rbp), %r14
  movq -0x20(%rbp), %rdi
  leaq -0x10(%rbp), %rsp
  pop %r14
  pop %rbx
  pop %rbp
  ret
hot_path:
  addq  $0x20, %rsp
  pop %r14
  pop %rbx
  pop %rbp
  ret
  .cfi_endproc
end:
  .size _start, .-_start

  .data
rel:  .quad end

# CHECK:   BOLT-INFO: Shrink wrapping moved 2 spills inserting load/stores and 0 spills inserting push/pops

# CHECK-OBJDUMP:     <_start>:
# CHECK-OBJDUMP:         leaq    (%rbp), %rsp
# CHECK-OBJDUMP-NEXT:    popq    %rbp
# CHECK-OBJDUMP-NEXT:    movq    -0x10(%rsp), %rbx
# CHECK-OBJDUMP-NEXT:    movq    -0x18(%rsp), %r14
# CHECK-OBJDUMP-NEXT:    retq
