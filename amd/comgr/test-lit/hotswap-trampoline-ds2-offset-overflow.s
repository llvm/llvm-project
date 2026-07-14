// COM: Test the offset-overflow guard in extractDsOperands. The single-
// COM: address DS instructions that the trampoline emits use a 16-bit
// COM: immediate offset field (max 0xFFFF = 65535 bytes). The stride64
// COM: forms scale a raw 8-bit per-operand index by (64 * ElemBytes), so
// COM: ds_*_2addr_stride64_b64 (Scale = 512) overflows for any raw index
// COM: >= 128:
// COM:
// COM:   raw 128 * 512 = 65536  -- one past the limit
// COM:   raw 255 * 512 = 130560 -- worst case
// COM:
// COM: When that happens the patch is not representable, so the trampoline
// COM: must leave the original (broken-on-A0) instruction in place rather
// COM: than emit a silently-truncated single-address replacement.
// COM:
// COM: Coverage:
// COM:   test_ds_load_b64_overflow : raw 128/255 -> scaled 65536/130560
// COM:                               (both off0 and off1 overflow)
// COM:   test_ds_load_b64_inrange  : raw 1/2 -> scaled 512/1024
// COM:                               (control: in-range stride64_b64 IS rewritten)

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// COM: Capture the verbose log on stderr to confirm the overflow message
// COM: fires for the out-of-range kernel. AMD_COMGR_EMIT_VERBOSE_LOGS=1
// COM: routes log() to llvm::errs(); we merge it into stdout for FileCheck.
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 \
// RUN:   hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=LOG %s
// COM: Pin the message shape (mnemonic, both raw + scaled values, the
// COM: 16-bit limit, and the "leaving original instruction in place"
// COM: closer) so a regression in the message format or the limit
// COM: constant fails here, not in some downstream-symptoms test. The
// COM: helper's specific message is the only diagnostic; patchDs2Addr does
// COM: not mislabel this intentional safe decline with a second generic
// COM: failure. RESULT: SUCCESS comes last because the rewrite as a whole
// COM: succeeds (the in-range kernel is patched).
// LOG:      hotswap: error: ds_load_2addr_stride64_b64 scaled offsets exceed
// LOG-SAME: the single-address DS 16-bit field
// LOG-SAME: off0=raw 128 * scale 512 = 65536
// LOG-SAME: off1=raw 255 * scale 512 = 130560
// LOG-SAME: max 65535
// LOG-SAME: leaving original instruction in place
// LOG:      RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// ---- Kernel 1: out-of-range -- patch must NOT fire --------------------------
// COM: Both per-operand indices scale past 0xFFFF (off0:128 -> 65536,
// COM: off1:255 -> 130560). The trampoline must reject the patch and
// COM: leave ds_load_2addr_stride64_b64 in the kernel verbatim. No
// COM: s_branch is inserted, no replacement ds_load_b64 appears, the
// COM: NOP sled is unused. The DISASM-NOT lines pin all three negatives.
// DISASM-LABEL: <test_ds_load_b64_overflow>:
// DISASM:       ds_load_2addr_stride64_b64 v[0:3], v4 offset0:128 offset1:255
// DISASM-NEXT:  s_wait_dscnt 0x0
// DISASM-NEXT:  s_endpgm
// DISASM-NOT:   s_branch
// DISASM-NOT:   ds_load_b64

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_ds_load_b64_overflow
.p2align 8
.type test_ds_load_b64_overflow,@function
test_ds_load_b64_overflow:
  ds_load_2addr_stride64_b64 v[0:3], v4 offset0:128 offset1:255
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
.Ltest_ds_load_b64_overflow_end:
.size test_ds_load_b64_overflow, .Ltest_ds_load_b64_overflow_end-test_ds_load_b64_overflow

// ---- Kernel 2: in-range -- patch MUST fire (negative control) --------------
// COM: Same opcode as kernel 1 but raw indices 1 and 2 (scaled 512 and
// COM: 1024, well below 0xFFFF). The patch must fire here -- otherwise
// COM: kernel 1 above would pass for the wrong reason (patch broken
// COM: across the board, not specifically gated by the overflow check).
// DISASM-LABEL: <test_ds_load_b64_inrange>:
// DISASM-NOT:   ds_load_2addr_stride64_b64
// DISASM:       s_branch
// DISASM:       s_wait_dscnt 0x0
// DISASM:       ds_load_b64 v[0:1], v4 offset:512
// DISASM-NEXT:  ds_load_b64 v[2:3], v4 offset:1024
// DISASM:       s_branch

.globl test_ds_load_b64_inrange
.p2align 8
.type test_ds_load_b64_inrange,@function
test_ds_load_b64_inrange:
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
.Ltest_ds_load_b64_inrange_end:
.size test_ds_load_b64_inrange, .Ltest_ds_load_b64_inrange_end-test_ds_load_b64_inrange

// COM: Idempotency. For the overflow kernel the original instruction is
// COM: still a ds_*_2addr_*, so the second pass would attempt to patch
// COM: it again, hit the same overflow, and again leave it alone. For
// COM: the in-range kernel the body now uses plain ds_load_b64, which
// COM: the dispatcher does not recognise, so it is also untouched. Net:
// COM: byte-identical output between passes.
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 \
// RUN:   hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.rodata
.p2align 8
.amdhsa_kernel test_ds_load_b64_overflow
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdhsa_kernel test_ds_load_b64_inrange
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel
