// Check that we add relevant linker flags for Android ARM/AArch64.

// RUN: %clang -### -target arm-linux-androideabi %s 2>&1 | \
// RUN:   FileCheck --check-prefix=MAX-PAGE-SIZE %s

// RUN: %clang -target aarch64-none-linux-android \
// RUN:   -### -v %s 2> %t
// RUN: FileCheck -check-prefix=GENERIC-ARM < %t %s

// RUN: %clang -target aarch64-none-linux-android \
// RUN:   -mcpu=cortex-a53 -### -v %s 2> %t
// RUN: FileCheck -check-prefix=CORTEX-A53 < %t %s

// RUN: %clang -target aarch64-none-linux-android \
// RUN:   -mcpu=cortex-a57 -### -v %s 2> %t
// RUN: FileCheck -check-prefix=CORTEX-A57 < %t %s

// RUN: %clang -target aarch64-none-linux-android \
// RUN:   -### -v %s 2> %t
// RUN: FileCheck -check-prefix=MAX-PAGE-SIZE < %t %s

// GENERIC-ARM: --fix-cortex-a53-843419
// CORTEX-A53: --fix-cortex-a53-843419
// CORTEX-A57-NOT: --fix-cortex-a53-843419
// MAX-PAGE-SIZE: "-z" "max-page-size=4096"
