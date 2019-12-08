// RUN: %clang -target amdgcn-amd-amdhsa -### -S -O3 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECKNONE %s
// RUN: %clang -target amdgcn-amd-amdhsa -### -S -O3 -fno-omit-frame-pointer %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECKALL %s
// RUN: %clang -target amdgcn-amd-amdhsa -### -S %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECKALL %s
// RUN: %clang -target amdgcn-amd-amdhsa -### -S -O0 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECKALL %s
// RUN: %clang -target amdgcn-amd-amdhsa -### -S -cl-opt-disable %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECKALL %s

// CHECKNONE: -mframe-pointer=none
// CHECKALL: -mframe-pointer=all
