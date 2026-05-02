// COM: Test HotSwap VOP3PX2 scale_src2 bit-field fix. V_WMMA_SCALE*
// COM: instructions have an unused scale_src2 field at bits [58:50] that
// COM: the SQ incorrectly decodes as an SGPR reference, causing a 3-cycle
// COM: SALU stall. The patch sets this field to VGPR0 encoding (0x100).
// COM: Applies to both A0 and B0 steppings.

// RUN: %clang --target=amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// COM: The V_WMMA_SCALE instruction must survive the rewrite; the patch
// COM: only modifies the scale_src2 bit-field, not the opcode or operands.
// COM: Verify the instruction is still present and decodable.
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM: v_wmma_scale_f32_16x16x128_f8f6f4
// DISASM: s_endpgm

// COM: Encoding-byte verification of the bit-field fix.
// COM: The assembler emits this 16-byte VOP3PX2 encoding for
// COM: v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7], v[8:23], v[24:35], v[40:47],
// COM: v1, v2 matrix_a_fmt:MATRIX_FMT_BF8 matrix_b_fmt:MATRIX_FMT_FP6
// COM: matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1:
// COM:   pre-patch:  00 08 35 cc 01 05 02 0a 00 08 33 cc 08 31 a2 14
// COM:                                    ^^ byte 7 = 0x0a
// COM:   post-patch: 00 08 35 cc 01 05 02 0c 00 08 33 cc 08 31 a2 14
// COM:                                    ^^ byte 7 = 0x0c
// COM: scale_src2 occupies bits [58:50] = byte 6 bits [7:2] | byte 7 bits
// COM: [2:0]. patchScaleSrc2 clears those bits and sets bit 2 of byte 7,
// COM: encoding VGPR0 (0x100). Byte 6's high six bits were already zero in
// COM: the assembler default; only byte 7 transitions 0x0a -> 0x0c (clear
// COM: bits [1:0], set bit 2). llvm-readelf groups bytes into 4-byte words
// COM: in byte order, so word 1 (bytes 4-7) reads 0105020c post-patch.
// RUN: %llvm-readelf -x .text %t.out.elf | %FileCheck --check-prefix=ENCODING %s
// ENCODING-LABEL: Hex dump of section '.text':
// ENCODING-NEXT: 0x{{[0-9a-f]+}} 000835cc 0105020c 000833cc 0831a214

// COM: Idempotency: the second rewrite must produce identical bytes.
// COM: patchScaleSrc2 returns false (no modification) on the second pass
// COM: only if the first pass already wrote the VGPR0 pattern, so cmp
// COM: passing is independent evidence that the bit-field is patched.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_vop3px2_src2
.p2align 8
.type test_vop3px2_src2,@function
test_vop3px2_src2:
  v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7], v[8:23], v[24:35], v[40:47], v1, v2 matrix_a_fmt:MATRIX_FMT_BF8 matrix_b_fmt:MATRIX_FMT_FP6 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
  s_endpgm
.Ltest_vop3px2_src2_end:
.size test_vop3px2_src2, .Ltest_vop3px2_src2_end-test_vop3px2_src2

.rodata
.p2align 8
.amdhsa_kernel test_vop3px2_src2
  .amdhsa_next_free_vgpr 48
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
