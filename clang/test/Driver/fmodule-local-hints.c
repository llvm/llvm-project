// RUN: %clang -### -c -fno-module-local-hints %s 2>&1 | FileCheck %s --check-prefix=OFF
// RUN: %clang -### -c -fmodule-local-hints -fno-module-local-hints %s 2>&1 | FileCheck %s --check-prefix=OFF
// RUN: %clang -### -c -fno-module-local-hints -fmodule-local-hints %s 2>&1 | FileCheck %s --check-prefix=ON
// RUN: %clang -### -c %s 2>&1 | FileCheck %s --check-prefix=ON

// OFF: "-cc1"
// OFF-SAME: "-fno-module-local-hints"
// ON-NOT: "-fmodule-local-hints"
// ON-NOT: "-fno-module-local-hints"
