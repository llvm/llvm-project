# RUN: llvm-mc -filetype=obj -triple amdgcn--amdhsa -mcpu=kaveri %s -o %t.o
# RUN: lld -flavor gnu %t.o -o %t
# RUN: llvm-readobj -sections -symbols -program-headers %t | FileCheck %s

# REQUIRES: amdgpu

.hsa_code_object_version 1,0
.hsa_code_object_isa 7,0,0,"AMD","AMDGPU"

.hsatext
.globl kernel0
.align 256
.amdgpu_hsa_kernel kernel0
kernel0:
  s_endpgm
.Lfunc_end0:
  .size kernel0, .Lfunc_end0-kernel0

.globl kernel1
.align 256
.amdgpu_hsa_kernel kernel1
kernel1:
  s_endpgm
  s_endpgm
.Lfunc_end1:
  .size kernel1, .Lfunc_end1-kernel1


# CHECK: Section {
# CHECK: Name: .hsatext
# CHECK: Type: SHT_PROGBITS
# CHECK: Flags [ (0xC00007)
# CHECK: SHF_ALLOC (0x2)
# CHECK: SHF_AMDGPU_HSA_AGENT (0x800000)
# CHECK: SHF_AMDGPU_HSA_CODE (0x400000)
# CHECK: SHF_EXECINSTR (0x4)
# CHECK: SHF_WRITE (0x1)
# CHECK: ]
# CHECK: Address: [[HSATEXT_ADDR:[0-9xa-f]+]]
# CHECK: }

# CHECK: Symbol {
# CHECK: Name: kernel0
# CHECK: Value: 0x0
# CHECK: Size: 4
# CHECK: Binding: Global
# CHECK: Type: AMDGPU_HSA_KERNEL
# CHECK: Section: .hsatext
# CHECK: }

# CHECK: Symbol {
# CHECK: Name: kernel1
# CHECK: Value: 0x100
# CHECK: Size: 8
# CHECK: Binding: Global
# CHECK: Type: AMDGPU_HSA_KERNEL
# CHECK: Section: .hsatext
# CHECK: }

# CHECK: ProgramHeader {
# CHECK: Type: PT_AMDGPU_HSA_LOAD_CODE_AGENT
# CHECK: VirtualAddress: [[HSATEXT_ADDR]]
# CHECK: }
