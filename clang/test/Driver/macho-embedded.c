// RUN: %clang -arch armv7 -target thumbv7-apple-darwin-eabi -### -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-AAPCS
// RUN: %clang -target x86_64-apple-macosx10.9 -arch armv7m -### -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-AAPCS
// RUN: %clang -arch armv7s -target thumbv7-apple-ios -### -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-APCS
// RUN: %clang -arch armv7s -target thumbv7-apple-darwin -### -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-APCS
// RUN: %clang -arch armv6m -target thumbv7-apple-darwin -### -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-MACHO-EMBEDDED
// RUN: %clang -arch armv7m -target thumbv7-apple-darwin -### -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-MACHO-EMBEDDED
// RUN: %clang -arch armv7em -target thumbv7-apple-darwin -### -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-MACHO-EMBEDDED

// RUN: %clang -arch armv7m --target=thumbv7-apple-ios -mios-version-min=5 -fdriver-only -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-MACHO-EMBEDDED-DIAG
// RUN: %clang -arch armv7m --target=thumbv7-apple-ios -mios-version-min=5 -mios-version-min=5 -fdriver-only -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-MACHO-EMBEDDED-DIAG
// RUN: %clang -arch armv7m --target=thumbv7-apple-watchos -mwatchos-version-min=5 -fdriver-only -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-MACHO-EMBEDDED-DIAG
// RUN: %clang -arch armv7m --target=thumbv7-apple-tvos -mtvos-version-min=5 -fdriver-only -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-MACHO-EMBEDDED-DIAG

// RUN: not %clang -arch armv7m --target=thumbv7-apple-tvos -mios-version-min=5 -### -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-MACHO-EMBEDDED-MIXED

// CHECK-IOS: "-triple" "thumbv7" "thumbv7-apple-ios

// CHECK-AAPCS: "-target-abi" "aapcs"
// CHECK-APCS: "-target-abi" "apcs-gnu"

// CHECK-MACHO-EMBEDDED: "-triple" "{{thumbv[67]e?m}}-apple-unknown-macho"
// CHECK-MACHO-EMBEDDED: "-mrelocation-model" "pic"
// CHECK-MACHO-EMBEDDED-DIAG: warning: argument unused during compilation: '-m{{ios|watchos|tvos}}-version-min=5'
// CHECK-MACHO-EMBEDDED-MIXED: error: unsupported option '-mios-version-min=' for target 'thumbv7-apple-tvos'
