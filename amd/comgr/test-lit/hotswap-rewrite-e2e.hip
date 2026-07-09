// REQUIRES: comgr-has-amdgpu-target
// COM: End-to-end coverage of the HotSwap rewrite pipeline on a real
// COM: clang-produced gfx1250 code object. Drives the full
// COM: compile -> hotswap-rewrite -> verify chain using the in-tree
// COM: clang + llvm-readelf + llvm-objdump substitutions.
// COM:
// COM: This kernel intentionally does not contain instructions matched by
// COM: the current hotswap patch set. The test is still useful as a real
// COM: code-object smoke test: it asserts that (1) rewrite returns SUCCESS,
// COM: (2) the output remains a valid gfx1250 ELF with preserved ISA
// COM: identification, and (3) .text is still present and disassemblable.
// COM: Patch-specific LIT tests cover opcode changes and trampolines.

// COM: Compile a tiny kernel targeting gfx1250 to a raw code object.
// RUN: %clang --offload-arch=gfx1250 --offload-device-only \
// RUN:     --no-gpu-bundle-output -nogpulib -nogpuinc \
// RUN:     %s -o %t.input.elf

// COM: Sanity: input is a gfx1250 ELF (attested in both the e_flags
// COM: field of the ELF header and the AMDHSA metadata note).
// RUN: %llvm-readelf -h %t.input.elf | %FileCheck --check-prefix=INPUT-FLAGS %s
// INPUT-FLAGS: Flags:{{.*}}gfx1250
// RUN: %llvm-readelf --notes %t.input.elf | %FileCheck --check-prefix=INPUT-META %s
// INPUT-META: amdhsa.target: amdgcn-amd-amdhsa{{--|-unknown-}}gfx1250

// COM: Run hotswap-rewrite; save the output so we can inspect it.
// RUN: hotswap-rewrite %t.input.elf \
// RUN:     amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:     --output %t.output.elf \
// RUN:   | %FileCheck --check-prefix=STATUS %s
// STATUS: RESULT: SUCCESS

// COM: Output ELF is valid and still identifies as gfx1250; this smoke
// COM: kernel has no patch-triggering instructions, so the rewrite should
// COM: preserve the target identification.
// RUN: %llvm-readelf -h %t.output.elf | %FileCheck --check-prefix=OUTPUT-FLAGS %s
// OUTPUT-FLAGS: Flags:{{.*}}gfx1250
// RUN: %llvm-readelf --notes %t.output.elf | %FileCheck --check-prefix=OUTPUT-META %s
// OUTPUT-META: amdhsa.target: amdgcn-amd-amdhsa{{--|-unknown-}}gfx1250

// COM: .text section is still present and marked executable.
// RUN: %llvm-readelf --section-headers %t.output.elf \
// RUN:   | %FileCheck --check-prefix=SECTIONS %s
// SECTIONS: .text{{.*}}PROGBITS

// COM: Disassembly is legible as AMDGPU code. The function symbol name
// COM: depends on C++ mangling; we do not hard-code it here.
// RUN: %llvm-objdump -d %t.output.elf | %FileCheck --check-prefix=DISASM %s
// DISASM: file format elf64-amdgpu
// DISASM: Disassembly of section .text:
// DISASM: s_endpgm

__attribute__((global))
void add(float *A, float *B, float *C) {
  *C = *A + *B;
}
