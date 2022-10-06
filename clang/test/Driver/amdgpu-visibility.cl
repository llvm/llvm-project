// RUN: %clang -### -target amdgcn-amd-amdhsa -x cl -c -emit-llvm %s 2>&1 | FileCheck -check-prefix=DEFAULT %s
// RUN: %clang -### -target amdgcn-amd-amdhsa -x cl -c -emit-llvm -fvisibility=protected  %s 2>&1 | FileCheck -check-prefix=OVERRIDE-PROTECTED  %s
// RUN: %clang -### -target amdgcn-amd-amdhsa -x cl -c -emit-llvm -fvisibility-ms-compat  %s 2>&1 | FileCheck -check-prefix=OVERRIDE-MS  %s

// RUN: %clang -### -target amdgcn-mesa-mesa3d -x cl -c -emit-llvm %s 2>&1 | FileCheck -check-prefix=DEFAULT %s
// RUN: %clang -### -target amdgcn-mesa-mesa3d -x cl -c -emit-llvm -fvisibility=protected  %s 2>&1 | FileCheck -check-prefix=OVERRIDE-PROTECTED  %s
// RUN: %clang -### -target amdgcn-mesa-mesa3d -x cl -c -emit-llvm -fvisibility-ms-compat  %s 2>&1 | FileCheck -check-prefix=OVERRIDE-MS  %s

// DEFAULT-DAG: "-fvisibility=hidden"
// DEFAULT-DAG: "-fapply-global-visibility-to-externs"

// OVERRIDE-PROTECTED-NOT: "-fapply-global-visibility-to-externs"
// OVERRIDE-PROTECTED: "-fvisibility=protected"
// OVERRIDE-PROTECTED-NOT: "-fapply-global-visibility-to-externs"

// OVERRIDE-MS-NOT: "-fapply-global-visibility-to-externs"
// OVERRIDE-MS-DAG: "-fvisibility=hidden"
// OVERRIDE-MS-DAG: "-ftype-visibility=default"
// OVERRIDE-MS-NOT: "-fapply-global-visibility-to-externs"
