// RUN: %clang -### -funique-basic-block-section-names %s -S 2>&1 | FileCheck -check-prefix=CHECK-OPT %s
// RUN: %clang -### -funique-basic-block-section-names -fno-unique-basic-block-section-names %s -S 2>&1 | FileCheck -check-prefix=CHECK-NOOPT %s
// CHECK-OPT: "-funique-basic-block-section-names"
// CHECK-NOOPT-NOT: "-funique-basic-block-section-names"
