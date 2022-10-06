// RUN: %clang -### -target x86_64-scei-ps4 %s -o - 2>&1 | \
// RUN:   FileCheck %s
// RUN: %clang -### -target x86_64-scei-ps4 -Wa,-mrelax-relocations=yes %s -o - 2>&1 | \
// RUN:   FileCheck %s
// RUN: %clang -### -target x86_64-scei-ps4 -Wa,-mrelax-relocations=no %s -o - 2>&1 | \
// RUN:   FileCheck -check-prefix=UNSET %s
// RUN: %clang -### -x assembler -target x86_64-scei-ps4 %s -o - 2>&1 | \
// RUN:   FileCheck %s
// RUN: %clang -### -x assembler -target x86_64-scei-ps4 -Wa,-mrelax-relocations=yes %s -o - 2>&1 | \
// RUN:   FileCheck %s
// RUN: %clang -### -x assembler -target x86_64-scei-ps4 -Wa,-mrelax-relocations=no %s -o - 2>&1 | \
// RUN:   FileCheck -check-prefix=UNSET %s

// RUN: %clang -### -target x86_64-sie-ps5 %s -o - 2>&1 | \
// RUN:   FileCheck %s
// RUN: %clang -### -target x86_64-sie-ps5 -Wa,-mrelax-relocations=yes %s -o - 2>&1 | \
// RUN:   FileCheck %s
// RUN: %clang -### -target x86_64-sie-ps5 -Wa,-mrelax-relocations=no %s -o - 2>&1 | \
// RUN:   FileCheck -check-prefix=UNSET %s
// RUN: %clang -### -x assembler -target x86_64-sie-ps5 %s -o - 2>&1 | \
// RUN:   FileCheck %s
// RUN: %clang -### -x assembler -target x86_64-sie-ps5 -Wa,-mrelax-relocations=yes %s -o - 2>&1 | \
// RUN:   FileCheck %s
// RUN: %clang -### -x assembler -target x86_64-sie-ps5 -Wa,-mrelax-relocations=no %s -o - 2>&1 | \
// RUN:   FileCheck -check-prefix=UNSET %s

// CHECK: "--mrelax-relocations"

// UNSET-NOT: "--mrelax-relocations"
