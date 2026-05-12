// Test HotSwap WMMA-split patches for GFX1250 B0-to-A0.
//
// The splitter replaces every K=128 fp8/bf8 WMMA with an s_branch into a
// trampoline at the tail of .text containing two K=64 halves followed by a
// branch back. The 32x16x128_f4 variant becomes two 16x16x128_f8f6f4 halves
// with both matrix-format modifiers forced to MATRIX_FMT_FP4. This test
// disassembles the patched ELF and checks that the original mnemonics are
// gone, the narrower replacement mnemonics appear in the trampoline region,
// and non-split instructions round-trip unchanged.
//
// Operand-shape note: every WMMA below uses register ranges where dst is
// disjoint from src0 and src1. The K-split second half is `WMMA dst,
// A_hi, B_hi, dst` -- if B_hi (the upper half of the original src1)
// overlapped dst, the second half would read B_hi from registers the
// first half just clobbered with the partial product. Compiler-generated
// WMMAs cannot land in that shape because the source pseudo carries
// `@earlyclobber $vdst` (VOP3PInstructions.td:1444), so the test inputs
// here mirror that contract -- any future change that breaks the slicing
// would be visible in the exact-operand DAGs at the bottom rather than
// being hidden by an incidental textual identity.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: Verify .text actually grew on the wire. Disassembly above shows the
// COM: replacement mnemonics, but a buggy rewriter could leave the .text
// COM: section header sh_size unchanged -- the disassembler walks raw bytes
// COM: regardless, but downstream tools that respect section headers (the
// COM: HSA loader's relocation pass, ELF strippers, debuggers) would then
// COM: miss the appended trampolines. Assert .text in the output is strictly
// COM: larger than .text in the input. Field 7 of llvm-readelf -S is the
// COM: hex Size column; the trailing space after `\.text` skips
// COM: `.text.<funcname>` would-be matches (none exist here, but cheap
// COM: insurance).
// COM: Drop `exit` from the awk one-liner: with `exit`, awk closes its
// COM: stdin before llvm-readelf finishes writing, and LIT's pipefail
// COM: shell propagates the SIGPIPE -> the test fails non-deterministically
// COM: in standalone runs (only passes in the bulk LIT run because
// COM: output buffering shifts the race). The `\.text ` pattern (with
// COM: trailing space) matches at most one section header per ELF, so
// COM: removing `exit` does not change the captured value.
// RUN: SIZE_IN=$(%llvm-readelf -S %t.elf | awk '/\.text /{print $7}') && \
// RUN:   SIZE_OUT=$(%llvm-readelf -S %t.out.elf | awk '/\.text /{print $7}') && \
// RUN:   test $((16#$SIZE_OUT)) -gt $((16#$SIZE_IN))

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"

// -- Test 1: 16x16x128_fp8_fp8 -> two 16x16x64_fp8_fp8 -----------------------
//
// DISASM-LABEL: <test_f32_16x16x128_fp8_fp8>:
// DISASM-NOT:   v_wmma_f32_16x16x128_fp8_fp8
// DISASM:       s_branch
.globl test_f32_16x16x128_fp8_fp8
.p2align 8
.type test_f32_16x16x128_fp8_fp8,@function
test_f32_16x16x128_fp8_fp8:
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], v[32:39]
  s_endpgm
.size test_f32_16x16x128_fp8_fp8, .-test_f32_16x16x128_fp8_fp8

// -- Test 2: 16x16x128_bf8_bf8 (f16 dest, 4-wide) -> two 16x16x64_bf8_bf8 ----
//
// DISASM-LABEL: <test_f16_16x16x128_bf8_bf8>:
// DISASM-NOT:   v_wmma_f16_16x16x128_bf8_bf8
// DISASM:       s_branch
.globl test_f16_16x16x128_bf8_bf8
.p2align 8
.type test_f16_16x16x128_bf8_bf8,@function
test_f16_16x16x128_bf8_bf8:
  v_wmma_f16_16x16x128_bf8_bf8 v[32:35], v[0:15], v[16:31], v[32:35]
  s_endpgm
.size test_f16_16x16x128_bf8_bf8, .-test_f16_16x16x128_bf8_bf8

// -- Test 3: 32x16x128_f4 -> two 16x16x128_f8f6f4 ----------------------------
//
// DISASM-LABEL: <test_f32_32x16x128_f4>:
// DISASM-NOT:   v_wmma_f32_32x16x128_f4
// DISASM:       s_branch
.globl test_f32_32x16x128_f4
.p2align 8
.type test_f32_32x16x128_f4,@function
test_f32_32x16x128_f4:
  v_wmma_f32_32x16x128_f4 v[32:47], v[0:15], v[16:23], v[32:47]
  s_endpgm
.size test_f32_32x16x128_f4, .-test_f32_32x16x128_f4

// -- Test 4: mixed-format 16x16x128_fp8_bf8 -> two 16x16x64_fp8_bf8 ----------
//
// DISASM-LABEL: <test_f32_16x16x128_fp8_bf8>:
// DISASM-NOT:   v_wmma_f32_16x16x128_fp8_bf8
// DISASM:       s_branch
.globl test_f32_16x16x128_fp8_bf8
.p2align 8
.type test_f32_16x16x128_fp8_bf8,@function
test_f32_16x16x128_fp8_bf8:
  v_wmma_f32_16x16x128_fp8_bf8 v[32:39], v[0:15], v[16:31], v[32:39]
  s_endpgm
.size test_f32_16x16x128_fp8_bf8, .-test_f32_16x16x128_fp8_bf8

// -- Test 5: 16x16x128_bf8_fp8 (f32) -> two 16x16x64_bf8_fp8 -----------------
//
// DISASM-LABEL: <test_f32_16x16x128_bf8_fp8>:
// DISASM-NOT:   v_wmma_f32_16x16x128_bf8_fp8
// DISASM:       s_branch
.globl test_f32_16x16x128_bf8_fp8
.p2align 8
.type test_f32_16x16x128_bf8_fp8,@function
test_f32_16x16x128_bf8_fp8:
  v_wmma_f32_16x16x128_bf8_fp8 v[32:39], v[0:15], v[16:31], v[32:39]
  s_endpgm
.size test_f32_16x16x128_bf8_fp8, .-test_f32_16x16x128_bf8_fp8

// -- Test 6: 16x16x128_bf8_bf8 (f32) -> two 16x16x64_bf8_bf8 -----------------
//
// DISASM-LABEL: <test_f32_16x16x128_bf8_bf8>:
// DISASM-NOT:   v_wmma_f32_16x16x128_bf8_bf8
// DISASM:       s_branch
.globl test_f32_16x16x128_bf8_bf8
.p2align 8
.type test_f32_16x16x128_bf8_bf8,@function
test_f32_16x16x128_bf8_bf8:
  v_wmma_f32_16x16x128_bf8_bf8 v[32:39], v[0:15], v[16:31], v[32:39]
  s_endpgm
.size test_f32_16x16x128_bf8_bf8, .-test_f32_16x16x128_bf8_bf8

// -- Test 7: 16x16x128_fp8_fp8 (f16 dest) -> two 16x16x64_fp8_fp8 ------------
//
// DISASM-LABEL: <test_f16_16x16x128_fp8_fp8>:
// DISASM-NOT:   v_wmma_f16_16x16x128_fp8_fp8
// DISASM:       s_branch
.globl test_f16_16x16x128_fp8_fp8
.p2align 8
.type test_f16_16x16x128_fp8_fp8,@function
test_f16_16x16x128_fp8_fp8:
  v_wmma_f16_16x16x128_fp8_fp8 v[32:35], v[0:15], v[16:31], v[32:35]
  s_endpgm
.size test_f16_16x16x128_fp8_fp8, .-test_f16_16x16x128_fp8_fp8

// -- Test 8: 16x16x128_fp8_bf8 (f16 dest) -> two 16x16x64_fp8_bf8 ------------
//
// DISASM-LABEL: <test_f16_16x16x128_fp8_bf8>:
// DISASM-NOT:   v_wmma_f16_16x16x128_fp8_bf8
// DISASM:       s_branch
.globl test_f16_16x16x128_fp8_bf8
.p2align 8
.type test_f16_16x16x128_fp8_bf8,@function
test_f16_16x16x128_fp8_bf8:
  v_wmma_f16_16x16x128_fp8_bf8 v[32:35], v[0:15], v[16:31], v[32:35]
  s_endpgm
.size test_f16_16x16x128_fp8_bf8, .-test_f16_16x16x128_fp8_bf8

// -- Test 9: 16x16x128_bf8_fp8 (f16 dest) -> two 16x16x64_bf8_fp8 ------------
//
// DISASM-LABEL: <test_f16_16x16x128_bf8_fp8>:
// DISASM-NOT:   v_wmma_f16_16x16x128_bf8_fp8
// DISASM:       s_branch
.globl test_f16_16x16x128_bf8_fp8
.p2align 8
.type test_f16_16x16x128_bf8_fp8,@function
test_f16_16x16x128_bf8_fp8:
  v_wmma_f16_16x16x128_bf8_fp8 v[32:35], v[0:15], v[16:31], v[32:35]
  s_endpgm
.size test_f16_16x16x128_bf8_fp8, .-test_f16_16x16x128_bf8_fp8

// -- Test 10: non-splittable instructions round-trip unchanged ---------------
//
// DISASM-LABEL: <test_no_split_required>:
// DISASM:       v_wmma_f32_16x16x32_f16
// DISASM:       v_add_f32
.globl test_no_split_required
.p2align 8
.type test_no_split_required,@function
test_no_split_required:
  v_wmma_f32_16x16x32_f16 v[32:39], v[0:7], v[8:15], v[32:39]
  v_add_f32_e32 v0, v1, v2
  s_endpgm
.size test_no_split_required, .-test_no_split_required

// -- Trampoline region: the splits land after the last original function. The
//    grown .text has no distinct symbol for the trampolines, so the
//    disassembly lists them under the <test_no_split_required> label
//    (anchored above). Assert each replacement mnemonic appears within that
//    region; CHECK-DAG lets the emission order change without breaking the
//    test. Eight K=64 fp8/bf8 replacement mnemonics (4 sign combinations x
//    {f16,f32} dest) plus the f4-split's f8f6f4 product cover the full
//    splitter table.
//
// COM: Exact register slicing for the fp8_fp8 K-split (input v[0:15],
// COM: v[16:31], v[32:39]). First half: A_lo=v[0:7], B_lo=v[16:23],
// COM: src2=original v[32:39]. Second half: A_hi=v[8:15], B_hi=v[24:31],
// COM: src2=dst v[32:39] (the carry from the first half). dst is unchanged
// COM: between halves. These two DAGs replace the bare-mnemonic check for
// COM: this opcode -- they're stricter and would catch off-by-one slicing
// COM: that a mnemonic-only check would miss.
// DISASM-DAG: v_wmma_f32_16x16x64_fp8_fp8 v[32:39], v[0:7], v[16:23], v[32:39]
// DISASM-DAG: v_wmma_f32_16x16x64_fp8_fp8 v[32:39], v[8:15], v[24:31], v[32:39]

// COM: Bare-mnemonic checks for the other 7 K-split products (one DAG
// COM: per opcode -- assignment to either the first-half or second-half
// COM: occurrence is unconstrained, which is fine because exact slicing
// COM: is verified via the fp8_fp8 case above).
// DISASM-DAG: v_wmma_f32_16x16x64_fp8_bf8
// DISASM-DAG: v_wmma_f32_16x16x64_bf8_fp8
// DISASM-DAG: v_wmma_f32_16x16x64_bf8_bf8
// DISASM-DAG: v_wmma_f16_16x16x64_fp8_fp8
// DISASM-DAG: v_wmma_f16_16x16x64_fp8_bf8
// DISASM-DAG: v_wmma_f16_16x16x64_bf8_fp8
// DISASM-DAG: v_wmma_f16_16x16x64_bf8_bf8

// COM: Exact register slicing for the M-split (input dst=v[32:47],
// COM: A=v[0:15], B=v[16:23], src2=v[32:47]). M is split in half: dst
// COM: and src2 each yield two 8-VGPR slices (v[32:39] for the first
// COM: half, v[40:47] for the second). A is split along M too (v[0:7]
// COM: then v[8:15]). B is broadcast (same v[16:23] on both halves).
// COM: The replacement opcode is v_wmma_f32_16x16x128_f8f6f4 with both
// COM: matrix-format modifiers literally MATRIX_FMT_FP4 so the f8f6f4
// COM: form interprets the data as f4 (matching the original opcode).
// DISASM-DAG: v_wmma_f32_16x16x128_f8f6f4 v[32:39], v[0:7], v[16:23], v[32:39]{{.*}}matrix_a_fmt:MATRIX_FMT_FP4{{.*}}matrix_b_fmt:MATRIX_FMT_FP4
// DISASM-DAG: v_wmma_f32_16x16x128_f8f6f4 v[40:47], v[8:15], v[16:23], v[40:47]{{.*}}matrix_a_fmt:MATRIX_FMT_FP4{{.*}}matrix_b_fmt:MATRIX_FMT_FP4

// Idempotency: rewriting the patched output again should produce identical
// bytes (the splitter only fires on K=128 mnemonics, which no longer exist
// in the rewritten ELF).
//
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf
