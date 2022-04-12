// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -### -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx908:xnack+:sramecc- \
// RUN:   -nostdlib %s 2>&1 | FileCheck %s

// RUN: %clang -### -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx908:xnack+:sramecc- \
// RUN:   -nostdlib -x ir %s 2>&1 | FileCheck %s

// RUN: %clang -### -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx908:xnack+:sramecc- \
// RUN:   -nostdlib -x assembler %s 2>&1 | FileCheck %s

// RUN: %clang -### -target amdgcn-amd-amdpal \
// RUN:   -mcpu=gfx908:xnack+:sramecc- \
// RUN:   -nostdlib %s 2>&1 | FileCheck %s

// RUN: %clang -### -target amdgcn--mesa3d \
// RUN:   -mcpu=gfx908:xnack+:sramecc- \
// RUN:   -nostdlib %s 2>&1 | FileCheck %s

// RUN: %clang -### -target amdgcn-amd-amdhsa \
// RUN:   -nostdlib %s 2>&1 | FileCheck -check-prefix=NONE %s

// CHECK: "-target-cpu" "gfx908"
// CHECK-SAME: "-target-feature" "-sramecc"
// CHECK-SAME: "-target-feature" "+xnack"

// NONE-NOT: "-target-cpu"
// NONE-NOT: "-target-feature"
