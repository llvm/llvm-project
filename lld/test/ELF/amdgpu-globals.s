# RUN: llvm-mc -filetype=obj -triple amdgcn--amdhsa -mcpu=kaveri %s -o %t.o
# RUN: lld -flavor gnu %t.o -o %t
# RUN: llvm-readobj -sections -symbols -program-headers %t | FileCheck %s

# REQUIRES: amdgpu

	.amdgpu_hsa_module_global module_global_program
	.size	module_global_program, 4
	.hsadata_global_program
module_global_program:
	.long	0                       ; 0x0

	.amdgpu_hsa_program_global program_global_program
	.size	program_global_program, 4
	.hsadata_global_program
program_global_program:
	.long	0                       ; 0x0

	.amdgpu_hsa_module_global module_global_agent
	.size	module_global_agent, 4
	.hsadata_global_agent
module_global_agent:
	.long	0                       ; 0x0

	.amdgpu_hsa_program_global program_global_agent
	.size	program_global_agent, 4
	.hsadata_global_agent
program_global_agent:
	.long	0                       ; 0x0

	.amdgpu_hsa_module_global module_global_readonly
	.size	module_global_readonly, 4
	.hsatext
module_global_readonly:
	.long	0                       ; 0x0

	.amdgpu_hsa_program_global program_global_readonly
	.size	program_global_readonly, 4
	.hsatext
program_global_readonly:
	.long	0                       ; 0x0

# CHECK: Section {
# CHECK: Name: .hsadata_global_program
# CHECK: Type: SHT_PROGBITS (0x1)
# CHECK: Flags [ (0x100003)
# CHECK: SHF_ALLOC (0x2)
# CHECK: SHF_AMDGPU_HSA_GLOBAL (0x100000)
# CHECK: SHF_WRITE (0x1)
# CHECK: ]
# CHECK: Address: [[HSADATA_GLOBAL_PROGRAM_ADDR:[0-9xa-f]+]]
# CHECK: }

# CHECK: Section {
# CHECK: Name: .hsadata_global_agent
# CHECK: Type: SHT_PROGBITS (0x1)
# CHECK: Flags [ (0x900003)
# CHECK: SHF_ALLOC (0x2)
# CHECK: SHF_AMDGPU_HSA_AGENT (0x800000)
# CHECK: SHF_AMDGPU_HSA_GLOBAL (0x100000)
# CHECK: SHF_WRITE (0x1)
# CHECK: ]
# CHECK: }

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
# CHECK: Name: module_global_agent
# CHECK: Value: 0x0
# CHECK: Size: 4
# CHECK: Binding: Local
# CHECK: Section: .hsadata_global_agent
# CHECK: }

# CHECK: Symbol {
# CHECK: Name: module_global_program
# CHECK: Value: 0x0
# CHECK: Size: 4
# CHECK: Binding: Local
# CHECK: Section: .hsadata_global_program
# CHECK: }

# CHECK: Symbol {
# CHECK: Name: module_global_readonly
# CHECK: Value: 0x0
# CHECK: Size: 4
# CHECK: Binding: Local
# CHECK: Type: Object
# CHECK: Section: .hsatext
# CHECK: }

# CHECK: Symbol {
# CHECK: Name: program_global_agent
# CHECK: Value: 0x4
# CHECK: Size: 4
# CHECK: Binding: Global
# CHECK: Type: Object
# CHECK: Section: .hsadata_global_agent
# CHECK: }

# CHECK: Symbol {
# CHECK: Name: program_global_program
# CHECK: Value: 0x4
# CHECK: Size: 4
# CHECK: Binding: Global
# CHECK: Type: Object
# CHECK: Section: .hsadata_global_program
# CHECK: }

# CHECK: Symbol {
# CHECK: Name: program_global_readonly
# CHECK: Value: 0x4
# CHECK: Size: 4
# CHECK: Binding: Global
# CHECK: Type: Object
# CHECK: Section: .hsatext
# CHECK: }

# CHECK: ProgramHeader {
# CHECK: Type: PT_AMDGPU_HSA_LOAD_GLOBAL_PROGRAM
# CHECK: VirtualAddress: [[HSADATA_GLOBAL_PROGRAM_ADDR]]
# CHECK: }

# CHECK: ProgramHeader {
# CHECK: Type: PT_AMDGPU_HSA_LOAD_CODE_AGENT
# CHECK: VirtualAddress: [[HSATEXT_ADDR]]
# CHECK: }
