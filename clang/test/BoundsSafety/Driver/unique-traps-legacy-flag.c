// Simple use of legacy flag
// RUN: %clang -fbounds-safety -funique-traps -### %s 2>&1 | FileCheck --check-prefix=POS %s
// RUN: %clang -fbounds-safety -fno-unique-traps -### %s 2>&1 | FileCheck --check-prefix=NEG %s

// Mix legacy flag with new flag
// RUN: %clang -fbounds-safety -funique-traps -fbounds-safety-unique-traps -### %s 2>&1 | FileCheck --check-prefix=POS %s
// RUN: %clang -fbounds-safety -fno-unique-traps -fbounds-safety-unique-traps  -### %s 2>&1 | FileCheck --check-prefix=NEG %s

// Last legacy flag wins for warning
// RUN: %clang -fbounds-safety -funique-traps -fno-unique-traps -funique-traps  -### %s 2>&1 | FileCheck --check-prefix=POS %s
// RUN: %clang -fbounds-safety -funique-traps -fno-unique-traps -### %s 2>&1 | FileCheck --check-prefix=NEG %s

// No warning
// RUN: %clang -fbounds-safety -fbounds-safety-unique-traps -### %s 2>&1 | FileCheck --check-prefix=NONE %s
// RUN: %clang -fbounds-safety -fno-bounds-safety-unique-traps -### %s 2>&1 | FileCheck --check-prefix=NONE %s
// RUN: %clang -fbounds-safety -### %s 2>&1 | FileCheck --check-prefix=NONE %s

// POS: warning: argument '-funique-traps' is deprecated, use 'fbounds-safety-unique-traps' instead [-Wdeprecated]
// NEG: warning: argument '-fno-unique-traps' is deprecated, use 'fno-bounds-safety-unique-traps' instead [-Wdeprecated]
// NONE-NOT: warning: argument '-f{{(no-)?}}unique-traps' is deprecated
