// OpenMP target arguments are parsed after the shared input translation.
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### /c \
// RUN:   -clang:-fopenmp -clang:-fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:   -clang:--offload-arch=gfx1100 -clang:-nogpulib \
// RUN:   -clang:-Xopenmp-target=amdgcn-amd-amdhsa -clang:-DFOO#7 -- %s 2>&1 | FileCheck %s
// RUN: %clang_cl --target=x86_64-unknown-linux-gnu -### /c \
// RUN:   -clang:-fopenmp -clang:-fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:   -clang:--offload-arch=gfx1100 -clang:-nogpulib \
// RUN:   -clang:-Xopenmp-target=amdgcn-amd-amdhsa -clang:-DFOO#7 -- %s 2>&1 | FileCheck %s
// RUN: %clang --target=x86_64-windows-msvc -### -c \
// RUN:   -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:   --offload-arch=gfx1100 -nogpulib \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -DFOO#7 -- %s 2>&1 | FileCheck %s
// CHECK: "-cc1" "-triple" "x86_64-{{[^"]*}}"
// CHECK-NOT: "FOO=7"
// CHECK: "-cc1" "-triple" "amdgpu11.00-amd-amdhsa"
// CHECK-SAME: "-D" "FOO=7"
// CHECK: "-cc1" "-triple" "x86_64-{{[^"]*}}"
// CHECK-NOT: "FOO=7"

// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### /c \
// RUN:   -clang:-fopenmp -clang:-fopenmp-targets=x86_64-pc-windows-msvc \
// RUN:   -clang:-nogpulib \
// RUN:   -clang:-Xopenmp-target=x86_64-pc-windows-msvc -clang:/O2 \
// RUN:   -clang:-Xopenmp-target=x86_64-pc-windows-msvc -clang:/permissive -- %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=DEVICE-O3

// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### /c \
// RUN:   -clang:-fopenmp -clang:-fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:   -clang:--offload-arch=gfx1100 -clang:-nogpulib \
// RUN:   -clang:-Xopenmp-target=amdgcn-amd-amdhsa -clang:/O2 \
// RUN:   -clang:-Xopenmp-target=amdgcn-amd-amdhsa -clang:/permissive -- %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=DEVICE-O3

// RUN: %clang_cl --target=x86_64-unknown-linux-gnu -### /c \
// RUN:   -clang:-fopenmp -clang:-fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:   -clang:--offload-arch=gfx1100 -clang:-nogpulib \
// RUN:   -clang:-Xopenmp-target=amdgcn-amd-amdhsa -clang:/O2 \
// RUN:   -clang:-Xopenmp-target=amdgcn-amd-amdhsa -clang:/permissive -- %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=DEVICE-O3

// DEVICE-O3: "-cc1" "-triple" "x86_64-{{[^"]*}}"
// DEVICE-O3-NOT: "-O3"
// DEVICE-O3-NOT: "-fno-operator-names"
// DEVICE-O3: "-cc1" "-triple" "{{[^"]*}}"
// DEVICE-O3-SAME: "-O3"
// DEVICE-O3-SAME: "-fno-operator-names"
// DEVICE-O3-SAME: "-fdelayed-template-parsing"
// DEVICE-O3: "-cc1" "-triple" "x86_64-{{[^"]*}}"
// DEVICE-O3-NOT: "-O3"
// DEVICE-O3-NOT: "-fno-operator-names"
// DEVICE-O3-NOT: argument unused during compilation

// A forwarded /Od must remove the other flags implied by shared /O2.
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### /c /O2 /permissive \
// RUN:   -clang:-fopenmp -clang:-fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:   -clang:--offload-arch=gfx1100 -clang:-nogpulib \
// RUN:   -clang:-Xopenmp-target=amdgcn-amd-amdhsa -clang:/Od \
// RUN:   -clang:-Xopenmp-target=amdgcn-amd-amdhsa -clang:/permissive- -- %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=DEVICE-OD
// DEVICE-OD: "-cc1" "-triple" "x86_64-pc-windows-msvc{{[^"]*}}"
// DEVICE-OD-SAME: "-O3"
// DEVICE-OD-SAME: "-fno-operator-names"
// DEVICE-OD: "-cc1" "-triple" "amdgpu11.00-amd-amdhsa"
// DEVICE-OD-NOT: "-fbuiltin"
// DEVICE-OD-NOT: "-ffunction-sections"
// DEVICE-OD: "-O0"
// DEVICE-OD-NOT: "-O3"
// DEVICE-OD-NOT: "-fno-operator-names"
// DEVICE-OD-NOT: "-fdelayed-template-parsing"
// DEVICE-OD: "-cc1" "-triple" "x86_64-pc-windows-msvc{{[^"]*}}"

// Repeated forwarded bundles use the last expandable option; explicit
// suboptions and canonical options still retain their parsed-list order.
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### /c \
// RUN:   -clang:-fopenmp -clang:-fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:   -clang:--offload-arch=gfx1100 -clang:-nogpulib \
// RUN:   -clang:-Xopenmp-target=amdgcn-amd-amdhsa -clang:/O2 \
// RUN:   -clang:-Xopenmp-target=amdgcn-amd-amdhsa -clang:/O1 \
// RUN:   -clang:-Xopenmp-target=amdgcn-amd-amdhsa -clang:/Ob0 \
// RUN:   -clang:-Xarch_device -clang:-O1 -- %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=DEVICE-NATIVE
// DEVICE-NATIVE: "-cc1" "-triple" "x86_64-pc-windows-msvc{{[^"]*}}"
// DEVICE-NATIVE: "-cc1" "-triple" "amdgpu11.00-amd-amdhsa"
// DEVICE-NATIVE-NOT: "-Os"
// DEVICE-NATIVE-NOT: "-O3"
// DEVICE-NATIVE: "-O1"
// DEVICE-NATIVE-SAME: "-fno-inline"
// DEVICE-NATIVE: "-cc1" "-triple" "x86_64-pc-windows-msvc{{[^"]*}}"

// GNU-mode Clang targeting MSVC also accepted these late slash options.
// RUN: %clang --target=x86_64-windows-msvc -### -c \
// RUN:   -fopenmp -fopenmp-targets=x86_64-pc-windows-msvc -nogpulib \
// RUN:   -Xopenmp-target=x86_64-pc-windows-msvc /O2 \
// RUN:   -Xopenmp-target=x86_64-pc-windows-msvc /permissive -- %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=DEVICE-O3

// Separate device architecture argument lists retain their own overrides.
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### /c /O2 \
// RUN:   -clang:-fopenmp -clang:-fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:   -clang:--offload-arch=gfx1100,gfx1101 -clang:-nogpulib \
// RUN:   -clang:-Xopenmp-target=amdgcn-amd-amdhsa -clang:/O1 \
// RUN:   -clang:-Xarch_gfx1100 -clang:-O0 -- %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=MULTIARCH
// MULTIARCH: "-cc1" "-triple" "x86_64-pc-windows-msvc{{[^"]*}}"
// MULTIARCH-SAME: "-O3"
// MULTIARCH: "-cc1" "-triple" "amdgpu11.00-amd-amdhsa"
// MULTIARCH-SAME: "-O0"
// MULTIARCH: "-cc1" "-triple" "amdgpu11.01-amd-amdhsa"
// MULTIARCH-SAME: "-Os"
// MULTIARCH: "-cc1" "-triple" "x86_64-pc-windows-msvc{{[^"]*}}"
