// UNSUPPORTED: system-windows

// Test that Debian-style multiarch layouts, in which libamdhip64.so is
// installed under <rocm>/lib/<multiarch-triple>/ rather than <rocm>/lib/, are
// detected by the driver and reflected in the linker command line.

// RUN: touch %t.o

// Inputs/rocm-multiarch mimics a Debian multiarch layout:
//   bin/.hipVersion signals a valid HIP install,
//   lib/x86_64-linux-gnu/libamdhip64.so is the runtime the driver must find.
// RUN: %clang -### --hip-link --target=x86_64-linux-gnu \
// RUN:   --rocm-path=%S/Inputs/rocm-multiarch %t.o 2>&1 \
// RUN:   | FileCheck -check-prefixes=MULTIARCH %s

// MULTIARCH: "{{.*/Inputs/rocm-multiarch/lib/x86_64-linux-gnu/libamdhip64.so}}"
// MULTIARCH: "-L{{.*/Inputs/rocm-multiarch/lib/x86_64-linux-gnu}}"

// Sanity check: when libamdhip64.so is *not* under lib/<multiarch-triple>/,
// the driver falls back to <rocm>/lib/ so non-Debian layouts are unchanged.
// RUN: %clang -### --hip-link --target=x86_64-linux-gnu \
// RUN:   --rocm-path=%S/Inputs/rocm %t.o 2>&1 \
// RUN:   | FileCheck -check-prefixes=FLAT %s

// FLAT: "{{.*/Inputs/rocm/lib/libamdhip64.so}}"
// FLAT-NOT: "{{.*/Inputs/rocm/lib/x86_64-linux-gnu/libamdhip64.so}}"
