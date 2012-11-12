// Check to make sure clang is somewhat picky about -g options.
// rdar://10383444

// RUN: %clang -### -c -g %s 2>&1 | FileCheck -check-prefix=G %s
// RUN: %clang -### -c -g2 %s 2>&1 | FileCheck -check-prefix=G %s
// RUN: %clang -### -c -g3 %s 2>&1 | FileCheck -check-prefix=G %s
// RUN: %clang -### -c -ggdb %s 2>&1 | FileCheck -check-prefix=G %s
// RUN: %clang -### -c -ggdb1 %s 2>&1 | FileCheck -check-prefix=G %s
// RUN: %clang -### -c -ggdb3 %s 2>&1 | FileCheck -check-prefix=G %s
// RUN: %clang -### -c -gdwarf-2 %s 2>&1 | FileCheck -check-prefix=G %s
//
// RUN: %clang -### -c -gfoo %s 2>&1 | FileCheck -check-prefix=G_NO %s
// RUN: %clang -### -c -g -g0 %s 2>&1 | FileCheck -check-prefix=G_NO %s
// RUN: %clang -### -c -ggdb0 %s 2>&1 | FileCheck -check-prefix=G_NO %s
//
// RUN: %clang -### -c -gline-tables-only %s 2>&1 \
// RUN:             | FileCheck -check-prefix=GLTO_ONLY %s
// RUN: %clang -### -c -gline-tables-only -g %s 2>&1 \
// RUN:             | FileCheck -check-prefix=G_ONLY %s
// RUN: %clang -### -c -gline-tables-only -g0 %s 2>&1 \
// RUN:             | FileCheck -check-prefix=GLTO_NO %s
//
// RUN: %clang -### -c -grecord-gcc-switches -gno-record-gcc-switches \
// RUN:           -gstrict-dwarf -gno-strict-dwarf %s 2>&1 \
// RUN:        | not grep "argument unused during compilation"
//
// G: "-cc1"
// G: "-g"
//
// G_NO: "-cc1"
// G_NO-NOT: "-g"
//
// GLTO_ONLY: "-cc1"
// GLTO_ONLY-NOT: "-g"
// GLTO_ONLY: "-gline-tables-only"
// GLTO_ONLY-NOT: "-g"
//
// G_ONLY: "-cc1"
// G_ONLY-NOT: "-gline-tables-only"
// G_ONLY: "-g"
// G_ONLY-NOT: "-gline-tables-only"
//
// GLTO_NO: "-cc1"
// GLTO_NO-NOT: "-gline-tables-only"
