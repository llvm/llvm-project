// COM: Splitting a compound DS instruction must preserve its read-before-write
// COM: semantics when the address or data registers overlap a destination.
// COM: This kernel deliberately has no nop sled, forcing the appended
// COM: trampoline/growth path. Its cyclic exchange is the intentional no-op
// COM: case: the original instruction must remain byte-stable.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

// DISASM-LABEL: <test_ds_overlap>:

// COM: A cyclic exchange dependency has no safe ordering and is declined.
// DISASM: ds_storexchg_2addr_rtn_b64 v[20:23], v24, v[22:23], v[20:21] offset1:1

// COM: Address overlaps the first b64 destination half: issue half 1 first.
// DISASM:      ds_load_b64 v[14:15], v12 offset:8
// DISASM-NEXT: ds_load_b64 v[12:13], v12

// COM: The same rule applies to b32.
// DISASM:      ds_load_b32 v5, v4 offset:8
// DISASM-NEXT: ds_load_b32 v4, v4 offset:4

// COM: Address overlaps the second half: natural order is already safe.
// DISASM:      ds_load_b64 v[16:17], v18
// DISASM-NEXT: ds_load_b64 v[18:19], v18 offset:8

// COM: Exchange op0 would clobber op1's address, so reverse the operations.
// DISASM:      ds_storexchg_rtn_b64 v[10:11], v8, v[14:15] offset:8
// DISASM-NEXT: ds_storexchg_rtn_b64 v[8:9], v8, v[12:13]

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_ds_overlap
.p2align 8
.type test_ds_overlap,@function
test_ds_overlap:
  ds_load_2addr_b64 v[12:15], v12 offset0:0 offset1:1
  ds_load_2addr_b32 v[4:5], v4 offset0:1 offset1:2
  ds_load_2addr_b64 v[16:19], v18 offset0:0 offset1:1
  ds_storexchg_2addr_rtn_b64 v[8:11], v8, v[12:13], v[14:15] offset0:0 offset1:1
  ds_storexchg_2addr_rtn_b64 v[20:23], v24, v[22:23], v[20:21] offset0:0 offset1:1
  s_wait_dscnt 0x0
  s_endpgm
.Ltest_ds_overlap_end:
.size test_ds_overlap, .Ltest_ds_overlap_end-test_ds_overlap

.rodata
.p2align 8
.amdhsa_kernel test_ds_overlap
  .amdhsa_next_free_vgpr 25
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
