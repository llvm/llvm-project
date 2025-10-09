// RUN: %clang -fbounds-safety -### %s 2>&1 | FileCheck --check-prefix=DISABLED %s

// Simple use of new flag
// RUN: %clang -fbounds-safety -fbounds-safety-unique-traps -### %s 2>&1 | FileCheck --check-prefix=ENABLED %s
// RUN: %clang -fbounds-safety -fno-bounds-safety-unique-traps -### %s 2>&1 | FileCheck --check-prefix=DISABLED %s

// New flag used multiple times
// RUN: %clang -fbounds-safety -fbounds-safety-unique-traps -fno-bounds-safety-unique-traps -### %s 2>&1 | FileCheck --check-prefix=DISABLED %s
// RUN: %clang -fbounds-safety -fno-bounds-safety-unique-traps -fbounds-safety-unique-traps -### %s 2>&1 | FileCheck --check-prefix=ENABLED %s
// RUN: %clang -fbounds-safety -fbounds-safety-unique-traps -fno-bounds-safety-unique-traps -fbounds-safety-unique-traps -### %s 2>&1 | FileCheck --check-prefix=ENABLED %s
// RUN: %clang -fbounds-safety -fno-bounds-safety-unique-traps -fbounds-safety-unique-traps -fno-bounds-safety-unique-traps -### %s 2>&1 | FileCheck --check-prefix=DISABLED %s

// Simple use of legacy flag
// RUN: %clang -fbounds-safety -fbounds-safety-unique-traps -### %s 2>&1 | FileCheck --check-prefix=ENABLED %s
// RUN: %clang -fbounds-safety -fno-unique-traps -### %s 2>&1 | FileCheck --check-prefix=DISABLED %s

// Legacy flag used multiple times
// RUN: %clang -fbounds-safety -funique-traps -fno-unique-traps -### %s 2>&1 | FileCheck --check-prefix=DISABLED %s
// RUN: %clang -fbounds-safety -fno-unique-traps -funique-traps -### %s 2>&1 | FileCheck --check-prefix=ENABLED %s
// RUN: %clang -fbounds-safety -funique-traps -fno-unique-traps -funique-traps -### %s 2>&1 | FileCheck --check-prefix=ENABLED %s
// RUN: %clang -fbounds-safety -fno-unique-traps -funique-traps -fno-unique-traps -### %s 2>&1 | FileCheck --check-prefix=DISABLED %s

// Mixed use of legacy and new flag
// RUN: %clang -fbounds-safety -funique-traps -fno-bounds-safety-unique-traps -### %s 2>&1 | FileCheck --check-prefix=DISABLED %s
// RUN: %clang -fbounds-safety -funique-traps -fbounds-safety-unique-traps -### %s 2>&1 | FileCheck --check-prefix=ENABLED %s

// RUN: %clang -fbounds-safety -fno-unique-traps -fno-bounds-safety-unique-traps -### %s 2>&1 | FileCheck --check-prefix=DISABLED %s
// RUN: %clang -fbounds-safety -fno-unique-traps -fbounds-safety-unique-traps -### %s 2>&1 | FileCheck --check-prefix=ENABLED %s

// RUN: %clang -fbounds-safety -fbounds-safety-unique-traps -fno-unique-traps -### %s 2>&1 | FileCheck --check-prefix=DISABLED %s
// RUN: %clang -fbounds-safety -fbounds-safety-unique-traps -funique-traps -### %s 2>&1 | FileCheck --check-prefix=ENABLED %s

// RUN: %clang -fbounds-safety -fno-bounds-safety-unique-traps -fno-unique-traps -### %s 2>&1 | FileCheck --check-prefix=DISABLED %s
// RUN: %clang -fbounds-safety -fno-bounds-safety-unique-traps -funique-traps -### %s 2>&1 | FileCheck --check-prefix=ENABLED %s


// ENABLED: -fbounds-safety-unique-traps
// DISABLED-NOT: -fbounds-safety-unique-traps
