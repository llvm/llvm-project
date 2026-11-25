// RUN: %clang --target=hexagon -### -g -c %s 2>&1 | FileCheck %s
// RUN: %clang --target=hexagon -### -g0 -g -c %s 2>&1 | FileCheck %s
// RUN: %clang --target=hexagon -### -c %s 2>&1 | FileCheck %s --check-prefix=NO
// RUN: %clang --target=hexagon -### -g -g0 -c %s 2>&1 | FileCheck %s --check-prefix=NO
// RUN: %clang --target=hexagon -### -g -ggdb0 -c %s 2>&1 | FileCheck %s --check-prefix=NO

// CHECK: "-mllvm" "-hexagon-save-parent-vsp"
// NO-NOT: "-hexagon-save-parent-vsp"
