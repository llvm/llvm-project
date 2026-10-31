// Shared source-language options must reach both HIP jobs.
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### /c \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /DFOO#7 /permissive- -- %s 2>&1 | FileCheck %s --check-prefix=COMMON

// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### /c \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /permissive -- %s 2>&1 | FileCheck %s --check-prefix=PERMISSIVE

// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### /c \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /DFOO#1 -clang:-Xarch_host -clang:-DFOO#2 -clang:-Xarch_device -clang:-DDEV#3 -- %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=SCOPED

// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### /c \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /permissive -clang:-Xarch_device -clang:-fno-delayed-template-parsing -clang:-Xarch_device \
// RUN:   -clang:-foperator-names -- %s 2>&1 | FileCheck %s --check-prefix=DEVICE-CONFORMING

// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### /c --no-offload-new-driver \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /DFOO#7 /permissive- -- %s 2>&1 | FileCheck %s --check-prefix=COMMON

// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### /c --no-offload-new-driver \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /permissive -- %s 2>&1 | FileCheck %s --check-prefix=PERMISSIVE

// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### /c --no-offload-new-driver \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /DFOO#1 -clang:-Xarch_host -clang:-DFOO#2 -clang:-Xarch_device -clang:-DDEV#3 -- %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=SCOPED

// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### /c --no-offload-new-driver \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /permissive -clang:-Xarch_device -clang:-fno-delayed-template-parsing -clang:-Xarch_device \
// RUN:   -clang:-foperator-names -- %s 2>&1 | FileCheck %s --check-prefix=DEVICE-CONFORMING

// RUN: %clang_cl --target=x86_64-unknown-linux-gnu -### /c \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /DFOO#7 /permissive- -- %s 2>&1 | FileCheck %s --check-prefix=COMMON

// RUN: %clang_cl --target=x86_64-unknown-linux-gnu -### /c \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /permissive -- %s 2>&1 | FileCheck %s --check-prefix=PERMISSIVE

// RUN: %clang_cl --target=x86_64-unknown-linux-gnu -### /c \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /DFOO#1 -clang:-Xarch_host -clang:-DFOO#2 -clang:-Xarch_device -clang:-DDEV#3 -- %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=SCOPED

// RUN: %clang_cl --target=x86_64-unknown-linux-gnu -### /c \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /permissive -clang:-Xarch_device -clang:-fno-delayed-template-parsing -clang:-Xarch_device \
// RUN:   -clang:-foperator-names -- %s 2>&1 | FileCheck %s --check-prefix=DEVICE-CONFORMING

// RUN: %clang_cl --target=x86_64-unknown-linux-gnu -### /c --no-offload-new-driver \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /DFOO#7 /permissive- -- %s 2>&1 | FileCheck %s --check-prefix=COMMON

// RUN: %clang_cl --target=x86_64-unknown-linux-gnu -### /c --no-offload-new-driver \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /permissive -- %s 2>&1 | FileCheck %s --check-prefix=PERMISSIVE

// RUN: %clang_cl --target=x86_64-unknown-linux-gnu -### /c --no-offload-new-driver \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /DFOO#1 -clang:-Xarch_host -clang:-DFOO#2 -clang:-Xarch_device -clang:-DDEV#3 -- %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=SCOPED

// RUN: %clang_cl --target=x86_64-unknown-linux-gnu -### /c --no-offload-new-driver \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /permissive -clang:-Xarch_device -clang:-fno-delayed-template-parsing -clang:-Xarch_device \
// RUN:   -clang:-foperator-names -- %s 2>&1 | FileCheck %s --check-prefix=DEVICE-CONFORMING

// COMMON: "-cc1" "-triple" "amdgpu11.00-amd-amdhsa"
// COMMON-SAME: "-D" "FOO=7"
// COMMON-NOT: "-fdelayed-template-parsing"
// COMMON-NOT: "-fno-operator-names"
// COMMON: "-cc1" "-triple" "x86_64-{{[^"]*}}"
// COMMON-SAME: "-D" "FOO=7"
// COMMON-NOT: "-fdelayed-template-parsing"
// COMMON-NOT: "-fno-operator-names"

// PERMISSIVE: "-cc1" "-triple" "amdgpu11.00-amd-amdhsa"
// PERMISSIVE-SAME: "-fno-operator-names"
// PERMISSIVE-SAME: "-fdelayed-template-parsing"
// PERMISSIVE: "-cc1" "-triple" "x86_64-{{[^"]*}}"
// PERMISSIVE-SAME: "-fno-operator-names"
// PERMISSIVE-SAME: "-fdelayed-template-parsing"

// SCOPED: "-cc1" "-triple" "amdgpu11.00-amd-amdhsa"
// SCOPED-SAME: "-D" "FOO=1" "-D" "DEV=3"
// SCOPED-NOT: "FOO=2"
// SCOPED: "-cc1" "-triple" "x86_64-{{[^"]*}}"
// SCOPED-SAME: "-D" "FOO=1" "-D" "FOO=2"
// SCOPED-NOT: "DEV=3"

// DEVICE-CONFORMING: "-cc1" "-triple" "amdgpu11.00-amd-amdhsa"
// DEVICE-CONFORMING-NOT: "-fno-operator-names"
// DEVICE-CONFORMING-NOT: "-fdelayed-template-parsing"
// DEVICE-CONFORMING: "-cc1" "-triple" "x86_64-{{[^"]*}}"
// DEVICE-CONFORMING-SAME: "-fno-operator-names"
// DEVICE-CONFORMING-SAME: "-fdelayed-template-parsing"

// GNU-mode offloading retains MSVC macro syntax when selected by the host.
// RUN: %clang --target=x86_64-pc-windows-msvc -### -c \
// RUN:   -x hip --offload-arch=gfx1100 -nogpuinc -nogpulib \
// RUN:   -DFOO#1 -Xarch_host -DFOO#2 -Xarch_device -DDEV#3 -- %s 2>&1 | FileCheck %s --check-prefix=SCOPED
