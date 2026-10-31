## Retained branches use the final target when it fits and the old
## entry trampoline otherwise.
# RUN: llvm-mc -triple=riscv64 -mattr=+c -filetype=obj %s -o %t.o
# RUN: ld.lld --emit-relocs --no-relax --gc-sections -Ttext=0x10fc0 -e _start %t.o -o %t.exe
# RUN: llvm-bolt %t.exe --skip-funcs=_start,branch_source --no-huge-pages --use-gnu-stack --align-text=4 --align-functions=4 --custom-allocation-vma=0x11000 -o %t.near
# RUN: llvm-objdump -d --mattr=+c -M no-aliases %t.near | FileCheck %s --check-prefix=NEAR
# RUN: llvm-bolt %t.exe --skip-funcs=_start,branch_source --no-huge-pages --use-gnu-stack --align-text=4 --align-functions=4 --custom-allocation-vma=0x400000 -o %t.far
# RUN: llvm-objdump -d --mattr=+c -M no-aliases %t.far | FileCheck %s --check-prefix=FAR

# NEAR-LABEL: <_start>:
# NEAR-NEXT: jal ra, {{.*}} <call_target>
# NEAR-LABEL: <branch_source>:
# NEAR-NEXT: bltu a1, a2, {{.*}} <branch_target>
# NEAR-NEXT: c.bnez a5, {{.*}} <branch_target>
# NEAR-NEXT: c.j {{.*}} <call_target>
# NEAR-NEXT: addi a3, a3, 0x7
# FAR-LABEL: <_start>:
# FAR-NEXT: jal ra, {{.*}} <call_target.org.0>
# FAR-LABEL: <branch_source>:
# FAR-NEXT: bltu a1, a2, {{.*}} <branch_target.org.0>
# FAR-NEXT: c.bnez a5, {{.*}} <branch_target.org.0>
# FAR-NEXT: c.j {{.*}} <call_target.org.0>

## A failed scan must discard both address patches and branch relocations.
## The separate call target can still move, so its old entry needs a trampoline.
# RUN: ld.lld --emit-relocs --no-relax --gc-sections -Ttext=0x10000 -e scan_mixed %t.o -o %t.mixed
# RUN: llvm-bolt %t.mixed --skip-funcs=scan_mixed --force-patch --no-huge-pages --use-gnu-stack --align-text=4 --align-functions=4 --custom-allocation-vma=0x20000 -o %t.mixed.bolt
# RUN: llvm-objdump -d %t.mixed.bolt | FileCheck %s --check-prefix=MIXED
# MIXED-LABEL: <scan_mixed>:
# MIXED-NEXT: auipc a1, 0x0
# MIXED-NEXT: jal {{.*}} <mixed_call_target.org.0>
# MIXED: addi a1, a1, 0x18
# MIXED-LABEL: 0000000000010018 <mixed_target>:
# MIXED-LABEL: 0000000000020000 <mixed_call_target>:

  .option norvc
  .option norelax
  .option exact
  .text
  .globl _start, call_target, branch_target, branch_source
  .type _start, @function
_start:
  jal ra, call_target
  jal ra, branch_source
  ret
  .size _start, .-_start

  .type call_target, @function
call_target:
  nop
  ret
  .size call_target, .-call_target

  .type branch_target, @function
branch_target:
  nop
  ret
  .size branch_target, .-branch_target

  .type branch_source, @function
branch_source:
  bltu a1, a2, branch_target
  .option push
  .option rvc
  c.bnez a5, branch_target
  c.j call_target
  .option pop
  addi a3, a3, 7
  ret
  .size branch_source, .-branch_source

  .section .text.scan_mixed,"ax",@progbits
  .globl scan_mixed, mixed_target, mixed_call_target
  .type scan_mixed, @function
scan_mixed:
.Lmixed_hi:
  auipc a1, %pcrel_hi(mixed_target)
  jal ra, mixed_call_target
  j .Lmixed_resume
  .insn 4, 0x0000000b
.Lmixed_resume:
  addi a1, a1, %pcrel_lo(.Lmixed_hi)
  ret
  .size scan_mixed, .-scan_mixed

  .type mixed_target, @function
mixed_target:
  ret
  nop
  .size mixed_target, .-mixed_target

  .type mixed_call_target, @function
mixed_call_target:
  ret
  nop
  .size mixed_call_target, .-mixed_call_target
