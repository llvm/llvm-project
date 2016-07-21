# RUN: llvm-mc -filetype=obj -triple=amdgcn--amdhsa -mcpu=fiji %s -o %t.o
# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-readobj -r %t.so | FileCheck %s
# RUN: llvm-objdump -s %t.so | FileCheck %s --check-prefix=OBJDUMP

# REQUIRES: amdgpu

  .text

kernel0:
  s_mov_b32 s0, common_var@GOTPCREL+4
  s_mov_b32 s0, extern_var@GOTPCREL+4
  s_mov_b32 s0, local_var+4
  s_mov_b32 s0, global_var@GOTPCREL+4
  s_mov_b32 s0, weak_var@GOTPCREL+4
  s_mov_b32 s0, weakref_var@GOTPCREL+4
  s_endpgm

  .comm   common_var,1024,4
  .globl  global_var
  .local  local_var
  .weak   weak_var
  .weakref weakref_var, weakref_alias_var

.section nonalloc, "w", @progbits
  .long var, common_var

# The relocation for local_var and var should be resolved by the linker.
# CHECK: Relocations [
# CHECK: .rela.dyn {
# CHECK-NEXT: R_AMDGPU_ABS64 common_var 0x0
# CHECK-NEXT: R_AMDGPU_ABS64 extern_var 0x0
# CHECK-NEXT: R_AMDGPU_ABS64 global_var 0x0
# CHECK-NEXT: R_AMDGPU_ABS64 weak_var 0x0
# CHECK-NEXT: R_AMDGPU_ABS64 weakref_alias_var 0x0
# CHECK-NEXT: }
# CHECK-NEXT: ]

# OBJDUMP: Contents of section nonalloc:
# OBJDUMP-NEXT: 0000 00000000 00300000
