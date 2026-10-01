// Check that clang-cl optimization options apply to both host and device.
// Device-specific options must still override the shared optimization level.

// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### /c --offload-new-driver \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /O1 -- %s 2>&1 | FileCheck %s --check-prefix=OS

// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### /c --offload-new-driver \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /O2 -- %s 2>&1 | FileCheck %s --check-prefix=O3

// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### /c --offload-new-driver \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /Ox -- %s 2>&1 | FileCheck %s --check-prefix=O3

// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### /c --offload-new-driver \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /Od -- %s 2>&1 | FileCheck %s --check-prefix=O0

// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### /c --offload-new-driver \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /O2 /Od -- %s 2>&1 | FileCheck %s --check-prefix=O0

// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### /c --offload-new-driver \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /Od /O2 -- %s 2>&1 | FileCheck %s --check-prefix=O3

// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### /c --offload-new-driver \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /O2 -clang:-O1 -- %s 2>&1 | FileCheck %s --check-prefix=O1

// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### /c --offload-new-driver \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /O2 -clang:-Xarch_device -clang:-O1 -- %s 2>&1 | FileCheck %s --check-prefix=DEVICE-O1

// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### /c --offload-new-driver \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /O2 /Ob0 -- %s 2>&1 | FileCheck %s --check-prefix=NOINLINE

// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### /c --no-offload-new-driver \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /O1 -- %s 2>&1 | FileCheck %s --check-prefix=OS

// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### /c --no-offload-new-driver \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /O2 -- %s 2>&1 | FileCheck %s --check-prefix=O3

// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### /c --no-offload-new-driver \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /Ox -- %s 2>&1 | FileCheck %s --check-prefix=O3

// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### /c --no-offload-new-driver \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /Od -- %s 2>&1 | FileCheck %s --check-prefix=O0

// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### /c --no-offload-new-driver \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /O2 /Od -- %s 2>&1 | FileCheck %s --check-prefix=O0

// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### /c --no-offload-new-driver \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /Od /O2 -- %s 2>&1 | FileCheck %s --check-prefix=O3

// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### /c --no-offload-new-driver \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /O2 -clang:-O1 -- %s 2>&1 | FileCheck %s --check-prefix=O1

// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### /c --no-offload-new-driver \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /O2 -clang:-Xarch_device -clang:-O1 -- %s 2>&1 | FileCheck %s --check-prefix=DEVICE-O1

// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### /c --no-offload-new-driver \
// RUN:   -x hip --offload-arch=gfx1100 -clang:-nogpuinc -clang:-nogpulib \
// RUN:   /O2 /Ob0 -- %s 2>&1 | FileCheck %s --check-prefix=NOINLINE

// OS: "-cc1" "-triple" "amdgpu11.00-amd-amdhsa"
// OS-SAME: "-Os"
// OS: "-cc1" "-triple" "x86_64-pc-windows-msvc{{[^"]*}}"
// OS-SAME: "-Os"

// O3: "-cc1" "-triple" "amdgpu11.00-amd-amdhsa"
// O3-SAME: "-O3"
// O3: "-cc1" "-triple" "x86_64-pc-windows-msvc{{[^"]*}}"
// O3-SAME: "-O3"

// O0: "-cc1" "-triple" "amdgpu11.00-amd-amdhsa"
// O0-SAME: "-O0"
// O0: "-cc1" "-triple" "x86_64-pc-windows-msvc{{[^"]*}}"
// O0-SAME: "-O0"

// O1: "-cc1" "-triple" "amdgpu11.00-amd-amdhsa"
// O1-SAME: "-O1"
// O1: "-cc1" "-triple" "x86_64-pc-windows-msvc{{[^"]*}}"
// O1-SAME: "-O1"

// DEVICE-O1: "-cc1" "-triple" "amdgpu11.00-amd-amdhsa"
// DEVICE-O1-SAME: "-O1"
// DEVICE-O1: "-cc1" "-triple" "x86_64-pc-windows-msvc{{[^"]*}}"
// DEVICE-O1-SAME: "-O3"

// NOINLINE: "-cc1" "-triple" "amdgpu11.00-amd-amdhsa"
// NOINLINE-SAME: "-O3"
// NOINLINE-SAME: "-fno-inline"
// NOINLINE: "-cc1" "-triple" "x86_64-pc-windows-msvc{{[^"]*}}"
// NOINLINE-SAME: "-O3"
// NOINLINE-SAME: "-fno-inline"
