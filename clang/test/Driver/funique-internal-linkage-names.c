// RUN: %clang -### -funique-internal-linkage-names %s -c 2>&1 | FileCheck -check-prefix=CHECK-OPT %s
// RUN: %clang -### -funique-internal-linkage-names -fno-unique-internal-linkage-names %s -c 2>&1 | FileCheck -check-prefix=CHECK-NOOPT %s
// CHECK-OPT: "-funique-internal-linkage-names"
// CHECK-NOOPT-NOT: "-funique-internal-linkage-names"
