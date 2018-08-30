// RUN: %clang -### -target amdgcn-amd-amdhsa -x cl -c -emit-llvm %s 2>&1 | FileCheck -check-prefix=DEFAULT %s
// RUN: %clang -### -target amdgcn-amd-amdhsa -x cl -c -emit-llvm -fvisibility=protected  %s 2>&1 | FileCheck -check-prefix=OVERRIDE-PROTECTED  %s
// RUN: %clang -### -target amdgcn-amd-amdhsa -x cl -c -emit-llvm -fvisibility-ms-compat  %s 2>&1 | FileCheck -check-prefix=OVERRIDE-MS  %s

// DEFAULT: "-fvisibility" "hidden"
// OVERRIDE-PROTECTED: "-fvisibility" "protected"
// OVERRIDE-MS:  "-fvisibility" "hidden" "-ftype-visibility" "default"
