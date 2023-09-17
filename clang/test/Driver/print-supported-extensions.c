// Test that --print-supported-extensions lists supported -march extensions
// on supported architectures, and errors on unsupported architectures.

// RUN: %if aarch64-registered-target %{ %clang --target=aarch64-linux-gnu \
// RUN:   --print-supported-extensions 2>&1 | FileCheck %s --check-prefix AARCH64 %}
// AARCH64: All available -march extensions for aarch64

// RUN: %if riscv-registered-target %{ %clang --target=riscv64-linux-gnu \
// RUN:   --print-supported-extensions 2>&1 | FileCheck %s --check-prefix RISCV %}
// RISCV: All available -march extensions for riscv64

// RUN: %if arm-registered-target %{ %clang --target=arm-linux-gnu \
// RUN:   --print-supported-extensions 2>&1 | FileCheck %s --check-prefix ARM %}
// ARM: All available -march extensions for arm

// RUN: %if x86-registered-target %{ %clang --target=x86_64-linux-gnu \
// RUN:   --print-supported-extensions 2>&1 | FileCheck %s --check-prefix X86 %}
// X86: All available -march extensions for x86-64
