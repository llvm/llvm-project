// COM: Test HotSwap rewrite API

// COM: Create a minimal test ELF file
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

// COM: Supported GFX1250 pair (stub returns input unchanged)
// RUN: hotswap-rewrite %t.elf amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   | %FileCheck --check-prefix=SUPPORTED %s
// SUPPORTED: RESULT: SUCCESS
