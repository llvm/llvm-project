/// Check that we report a warning instead of an error for target-specific compilation only options.
// RUN: %clang -### --target=aarch64 -faddrsig -mbranch-protection=standard -c %s 2>&1 | FileCheck %s
// RUN: %clang -### --target=aarch64 -faddrsig -mbranch-protection=standard -c -fno-integrated-as %s 2>&1 | FileCheck %s

/// Report a warning if we perform the link phase.
// RUN: %clang -### --target=aarch64 -faddrsig -mbranch-protection=standard %s 2>&1 | FileCheck %s

// CHECK: warning: argument unused during compilation: '-faddrsig'
// CHECK: warning: argument unused during compilation: '-mbranch-protection=standard'

/// assembler-with-cpp claims compile only options. Ideally we should emit a warning.
// RUN: %clang -### -Werror --target=aarch64 -c -faddrsig -mbranch-protection=standard -x assembler-with-cpp %s
