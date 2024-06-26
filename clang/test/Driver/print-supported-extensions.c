// Test that --print-supported-extensions lists supported -march extensions
// on supported architectures, and errors on unsupported architectures.

// RUN: %if aarch64-registered-target %{ %clang --target=aarch64-linux-gnu \
// RUN:   --print-supported-extensions 2>&1 | FileCheck %s --check-prefix AARCH64 %}
// AARCH64: All available -march extensions for AArch64
// AARCH64:     Name                Description
// AARCH64:     aes                 Enable AES support (FEAT_AES, FEAT_PMULL)

// RUN: %if riscv-registered-target %{ %clang --target=riscv64-linux-gnu \
// RUN:   --print-supported-extensions 2>&1 | FileCheck %s --check-prefix RISCV %}
// RISCV: All available -march extensions for RISC-V
// RISCV:     Name                Version   Description
// RISCV:     i                   2.1

// RUN: %if arm-registered-target %{ %clang --target=arm-linux-gnu \
// RUN:   --print-supported-extensions 2>&1 | FileCheck %s --check-prefix ARM %}
// ARM: All available -march extensions for ARM
// ARM:     Name                Description
// ARM:     crc                 Enable support for CRC instructions

// RUN: %if x86-registered-target %{ not %clang --target=x86_64-linux-gnu \
// RUN:   --print-supported-extensions 2>&1 | FileCheck %s --check-prefix X86 %}
// X86: error: option '--print-supported-extensions' cannot be specified on this target