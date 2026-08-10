// RUN: %clang --target=aarch64-none-gnu --verbose -mcpu= -march= 2>&1 | FileCheck %s --check-prefix=WARNING
// RUN: %clang --target=aarch64-none-gnu -### -mcpu= -march= 2>&1 | FileCheck %s --check-prefix=WARNING

// RUN: %clang --target=x86_64-unknown-linux-gnu --verbose -mcpu= -march= 2>&1 | FileCheck %s --check-prefix=WARNING
// RUN: %clang --target=x86_64-unknown-linux-gnu -### -mcpu= -march= 2>&1 | FileCheck %s --check-prefix=WARNING

/// In situation when there is no compilation/linking clang should not emit error
/// about target specific options, but just warn that are not used.
WARNING: warning: argument unused during compilation
WARNING: warning: argument unused during compilation
