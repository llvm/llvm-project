// REQUIRES: zlib, amdgpu-registered-target

// RUN: %clang -### -target amdgcn-amd-amdhsa -gz=none -x assembler %s 2>&1 | FileCheck -check-prefix CHECK-OPT_GZ_EQ_NONE %s
// RUN: %clang -### -target amdgcn-amd-amdhsa -gz=none -nogpulib %s 2>&1 | FileCheck -check-prefix CHECK-OPT_GZ_EQ_NONE %s
// CHECK-OPT_GZ_EQ_NONE: {{.* "-cc1(as)?".* "--compress-debug-sections=none"}}
// CHECK-OPT_GZ_EQ_NONE: "--compress-debug-sections=none"

// RUN: %clang -### -target amdgcn-amd-amdhsa -gz=zlib -x assembler %s 2>&1 | FileCheck -check-prefix CHECK-OPT_GZ_EQ_ZLIB %s
// RUN: %clang -### -target amdgcn-amd-amdhsa -gz=zlib -nogpulib %s 2>&1 | FileCheck -check-prefix CHECK-OPT_GZ_EQ_ZLIB %s
// CHECK-OPT_GZ_EQ_ZLIB: {{.* "-cc1(as)?".* "--compress-debug-sections=zlib"}}
// CHECK-OPT_GZ_EQ_ZLIB: "--compress-debug-sections=zlib"
