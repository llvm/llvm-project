// Ensure that the driver flag is propagated to cc1.
// RUN: %clang -### -fc++-abi=itanium -target x86_64-unknown-linux-gnu %s \
// RUN:   2>&1 | FileCheck %s -check-prefix=ITANIUM
// RUN: %clang -### -fc++-abi=fuchsia -target x86_64-unknown-fuchsia %s \
// RUN:   2>&1 | FileCheck %s -check-prefix=FUCHSIA
// RUN: %clang -### -fc++-abi=microsoft -target x86_64-unknown-windows-msvc %s \
// RUN:   2>&1 | FileCheck %s -check-prefix=MICROSOFT

// ITANIUM: -fc++-abi=itanium
// FUCHSIA: -fc++-abi=fuchsia
// MICROSOFT: -fc++-abi=microsoft
