// COM: Test HotSwap trampoline patch: ds_*_2addr_stride64_* expansion into
// COM: two single-address DS instructions. Each kernel here uses a drain
// COM: s_wait_dscnt 0x0, which must stay at 0x0 after splitting (see the
// COM: bumpNextWaitDscnt header for the rationale).
// COM:
// COM: Covers b32 load, b64 load, b32 store, and b32 exchange operand
// COM: variants via the NOP sled emission mechanism. Verifies explicit
// COM: s_branch generation for the forward/back jumps.
// COM:
// COM: Companion tests:
// COM:   hotswap-trampoline-ds-multi.s     -- drain preservation under stacking
// COM:   hotswap-trampoline-ds-pipelined.s -- non-drain bump path (0x1 -> 0x2/0x3)
// COM:   hotswap-trampoline-ds-nosled.s    -- true trampoline fallback (no NOP sled)
// COM:   hotswap-trampoline-ds-nowait.s    -- control-flow guard (no s_wait_dscnt)

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: --- Per-kernel checks ---

// COM: Kernel 1 (b32 load): s_branch forward to sled, the wait stays at the
// COM: original position with imm unchanged (0x0), expanded loads appear in
// COM: the sled area with s_branch back to the wait instruction.
// DISASM-LABEL: <test_ds_load_b32>:
// DISASM-NOT: ds_load_2addr_stride64_b32
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x0
// DISASM: ds_load_b32 v0
// DISASM: ds_load_b32 v1
// DISASM: s_branch

// COM: Kernel 2 (b64 load): b64 register pairs formatted as v[X:Y].
// DISASM-LABEL: <test_ds_load_b64>:
// DISASM-NOT: ds_load_2addr_stride64_b64
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x0
// DISASM: ds_load_b64 v[0:1]
// DISASM: ds_load_b64 v[2:3]
// DISASM: s_branch

// COM: Kernel 3 (b32 store): store operand layout (addr, data0, data1).
// DISASM-LABEL: <test_ds_store_b32>:
// DISASM-NOT: ds_store_2addr_stride64_b32
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x0
// DISASM: ds_store_b32 v2, v0
// DISASM: ds_store_b32 v2, v1
// DISASM: s_branch

// COM: Kernel 4 (b32 exchange): exchange operand layout (dst, addr, data).
// DISASM-LABEL: <test_ds_xchg_b32>:
// DISASM-NOT: ds_storexchg_2addr_stride64_rtn_b32
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x0
// DISASM: ds_storexchg_rtn_b32 v0
// DISASM: ds_storexchg_rtn_b32 v1
// DISASM: s_branch

// COM: Idempotency: rewriting the output again should produce identical bytes.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

// ---- Kernel 1: ds_load_2addr_stride64_b32 (base case) -----------------------

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_ds_load_b32
.p2align 8
.type test_ds_load_b32,@function
test_ds_load_b32:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
  s_wait_dscnt 0x0
  s_endpgm
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.Ltest_ds_load_b32_end:
.size test_ds_load_b32, .Ltest_ds_load_b32_end-test_ds_load_b32

// ---- Kernel 2: ds_load_2addr_stride64_b64 (b64 element size) ----------------

.globl test_ds_load_b64
.p2align 8
.type test_ds_load_b64,@function
test_ds_load_b64:
  ds_load_2addr_stride64_b64 v[0:3], v4 offset0:1 offset1:2
  s_wait_dscnt 0x0
  s_endpgm
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.Ltest_ds_load_b64_end:
.size test_ds_load_b64, .Ltest_ds_load_b64_end-test_ds_load_b64

// ---- Kernel 3: ds_store_2addr_stride64_b32 (store operand layout) -----------

.globl test_ds_store_b32
.p2align 8
.type test_ds_store_b32,@function
test_ds_store_b32:
  ds_store_2addr_stride64_b32 v2, v0, v1 offset0:1 offset1:3
  s_wait_dscnt 0x0
  s_endpgm
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.Ltest_ds_store_b32_end:
.size test_ds_store_b32, .Ltest_ds_store_b32_end-test_ds_store_b32

// ---- Kernel 4: ds_storexchg_2addr_stride64_rtn_b32 (exchange layout) --------

.globl test_ds_xchg_b32
.p2align 8
.type test_ds_xchg_b32,@function
test_ds_xchg_b32:
  ds_storexchg_2addr_stride64_rtn_b32 v[0:1], v2, v3, v4 offset0:1 offset1:3
  s_wait_dscnt 0x0
  s_endpgm
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.Ltest_ds_xchg_b32_end:
.size test_ds_xchg_b32, .Ltest_ds_xchg_b32_end-test_ds_xchg_b32

.rodata
.p2align 8
.amdhsa_kernel test_ds_load_b32
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdhsa_kernel test_ds_load_b64
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdhsa_kernel test_ds_store_b32
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdhsa_kernel test_ds_xchg_b32
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel
