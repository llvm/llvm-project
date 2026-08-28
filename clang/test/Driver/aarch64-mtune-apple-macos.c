// Test -mtune flag on Apple Silicon AArch64 macOS.

// Test default -mtune on macOS
// RUN: %clang -### -c --target=arm64-apple-macos %s 2>&1 | FileCheck %s --check-prefixes=CPU-MACOS-DEFAULT,TUNE-MACOS-DEFAULT

// RUN: %clang -### -c --target=arm64-apple-macos %s -mtune=generic 2>&1 | FileCheck %s --check-prefixes=CPU-MACOS-DEFAULT,TUNE-GENERIC

// RUN: %clang -### -c --target=arm64-apple-macos %s -mtune=apple-m5 2>&1 | FileCheck %s --check-prefixes=CPU-MACOS-DEFAULT,TUNE-APPLE-M5

// Check interaction between march and mtune.

// RUN: %clang -### -c --target=arm64-apple-macos %s -march=armv8-a 2>&1 | FileCheck %s --check-prefixes=CPU-MACOS-DEFAULT,TUNE-MACOS-DEFAULT

// RUN: %clang -### -c --target=arm64-apple-macos %s -march=armv8-a -mtune=apple-m5 2>&1 | FileCheck %s --check-prefixes=CPU-MACOS-DEFAULT,TUNE-APPLE-M5

// Check interaction between mcpu and mtune.

// RUN: %clang -### -c --target=arm64-apple-macos %s -mcpu=apple-m4 2>&1 | FileCheck %s --check-prefixes=CPU-APPLE-M4,NO-TUNE

// RUN: %clang -### -c --target=arm64-apple-macos %s -mcpu=apple-m4 -mtune=apple-m5 2>&1 | FileCheck %s --check-prefixes=CPU-APPLE-M4,TUNE-APPLE-M5

// CPU-MACOS-DEFAULT: "-target-cpu" "apple-m1"
// CPU-APPLE-M4: "-target-cpu" "apple-m4"

// TUNE-MACOS-DEFAULT: "-tune-cpu" "apple-m5"
// TUNE-APPLE-M5: "-tune-cpu" "apple-m5"
// TUNE-GENERIC: "-tune-cpu" "generic"
// NO-TUNE-NOT: "-tune-cpu"
