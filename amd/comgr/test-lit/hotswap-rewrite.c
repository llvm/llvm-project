// COM: Test HotSwap rewrite API

// COM: Create a minimal test ELF file (ELF64 header only, no sections).
// RUN: printf '\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00' > %t.elf

// COM: NULL-argument validation (no args)
// RUN: hotswap-rewrite | %FileCheck --check-prefix=NULL %s
// NULL: NULL_ARGS: INVALID_ARGUMENT

// COM: Unsupported ISA pair
// RUN: hotswap-rewrite %t.elf amdgcn-amd-amdhsa--gfx942 amdgcn-amd-amdhsa--gfx942 \
// RUN:   | %FileCheck --check-prefix=INVALID %s
// INVALID: RESULT: INVALID_ARGUMENT

// COM: Invalid ISA string
// RUN: hotswap-rewrite %t.elf not-a-valid-isa also-not-valid \
// RUN:   | %FileCheck --check-prefix=BADISA %s
// BADISA: RESULT: INVALID_ARGUMENT

// COM: Zero-size input with supported ISA
// RUN: hotswap-rewrite %t.elf amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 --zero-size \
// RUN:   | %FileCheck --check-prefix=ZEROSIZE %s
// ZEROSIZE: RESULT: INVALID_ARGUMENT

// COM: Supported GFX1250 pair on a malformed ELF (no .text section).
// COM: retargetCodeObjectB0A0 rejects inputs that fail ELF64 parsing or have
// COM: an empty .text section with INVALID_ARGUMENT -- returning SUCCESS with
// COM: an unchanged copy there would silently hide caller-side bugs.
// RUN: hotswap-rewrite %t.elf amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   | %FileCheck --check-prefix=MALFORMED %s
// MALFORMED: RESULT: INVALID_ARGUMENT

// COM: End-to-end coverage on a real gfx1250 code object (compiled via clang
// COM: --offload-arch=gfx1250, verified with llvm-readelf + llvm-objdump) is
// COM: tracked as a follow-up once the gfx1250 kernel-compile driver is wired
// COM: into the test-lit infrastructure.
