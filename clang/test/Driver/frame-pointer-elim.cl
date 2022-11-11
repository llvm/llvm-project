// RUN: %clang --target=amdgcn-amd-amdhsa -### -S -O3 %s 2>&1 | FileCheck -check-prefix=CHECKNONE %s
// RUN: %clang --target=amdgcn-amd-amdhsa -### -S -O3 -fno-omit-frame-pointer %s 2>&1 | FileCheck -check-prefix=CHECKALL %s
// RUN: %clang --target=amdgcn-amd-amdhsa -### -S %s 2>&1 | FileCheck -check-prefix=CHECKALL %s
// RUN: %clang --target=amdgcn-amd-amdhsa -### -S -O0 %s 2>&1 | FileCheck -check-prefix=CHECKALL %s
// RUN: %clang --target=amdgcn-amd-amdhsa -### -S -cl-opt-disable %s 2>&1 | FileCheck -check-prefix=CHECKALL %s

// CHECKNONE: -mframe-pointer=none
// CHECKALL: -mframe-pointer=all
