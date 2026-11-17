// RUN: %clang -### -c -fmodule-local-hints %s 2>&1 | FileCheck %s --check-prefix=ON
// RUN: %clang -### -c -fno-module-local-hints -fmodule-local-hints %s 2>&1 | FileCheck %s --check-prefix=ON
// RUN: %clang -### -c -fmodule-local-hints -fno-module-local-hints %s 2>&1 | FileCheck %s --check-prefix=OFF
// RUN: %clang -### -c %s 2>&1 | FileCheck %s --check-prefix=OFF

// ON: "-cc1"
// ON-SAME: "-fmodule-local-hints"
// OFF-NOT: "-fmodule-local-hints"
// OFF-NOT: "-fno-module-local-hints"
