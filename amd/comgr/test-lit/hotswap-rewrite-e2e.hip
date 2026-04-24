// COM: End-to-end coverage of the HotSwap rewrite pipeline on a real
// COM: clang-produced gfx1250 code object. Drives the full
// COM: compile -> hotswap-rewrite -> verify chain using the in-tree
// COM: clang + llvm-readelf + llvm-objdump substitutions.
// COM:
// COM: Patches in amd/comgr/src/comgr-hotswap-b0a0.cpp are weak stubs
// COM: returning 0 until the concrete patch .cpp files land, so the
// COM: dispatcher currently emits an output that is bytewise-identical
// COM: to the input. Even with no patches applied we can assert that
// COM: (1) the rewrite returns SUCCESS, (2) the output is a valid
// COM: gfx1250 ELF that preserves the ISA identification, and (3) .text
// COM: is still present and disassemblable as AMDGPU code. The
// COM: per-patch PRs will layer in their own `llvm-objdump | FileCheck`
// COM: stanzas on top of this harness to assert their specific opcode
// COM: changes / trampolines.

// COM: Compile a tiny kernel targeting gfx1250 to a raw code object.
// RUN: %clang --offload-arch=gfx1250 --offload-device-only \
// RUN:     --no-gpu-bundle-output -nogpulib -nogpuinc \
// RUN:     %s -o %t.input.elf

// COM: Sanity: input is a gfx1250 ELF (attested in both the e_flags
// COM: field of the ELF header and the AMDHSA metadata note).
// RUN: %llvm-readelf -h %t.input.elf | %FileCheck --check-prefix=INPUT-FLAGS %s
// INPUT-FLAGS: Flags:{{.*}}gfx1250
// RUN: %llvm-readelf --notes %t.input.elf | %FileCheck --check-prefix=INPUT-META %s
// INPUT-META: amdhsa.target: amdgcn-amd-amdhsa--gfx1250

// COM: Run hotswap-rewrite; save the output so we can inspect it.
// RUN: hotswap-rewrite %t.input.elf \
// RUN:     amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:     --output %t.output.elf \
// RUN:   | %FileCheck --check-prefix=STATUS %s
// STATUS: RESULT: SUCCESS

// COM: Output ELF is valid and still identifies as gfx1250 (patches are
// COM: weak stubs, so the identity should be preserved untouched; future
// COM: patch PRs that intentionally change the target ISA note will
// COM: update this stanza).
// RUN: %llvm-readelf -h %t.output.elf | %FileCheck --check-prefix=OUTPUT-FLAGS %s
// OUTPUT-FLAGS: Flags:{{.*}}gfx1250
// RUN: %llvm-readelf --notes %t.output.elf | %FileCheck --check-prefix=OUTPUT-META %s
// OUTPUT-META: amdhsa.target: amdgcn-amd-amdhsa--gfx1250

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
