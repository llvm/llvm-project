// REQUIRES: default-pie

// -fPIC and -fPIE only affects host relocation model.
// device compilation always uses PIC.

// RUN: %clang -### --target=x86_64-unknown-linux-gnu \
// RUN:   --offload-arch=gfx906 %s -nogpulib -nogpuinc \
// RUN:   2>&1 | FileCheck -check-prefixes=DEV,HOST-PIE \
// RUN:   -DHOST_TRIPLE=x86_64-unknown-linux-gnu %s

// RUN: %clang -### --target=x86_64-unknown-linux-gnu \
// RUN:   -fgpu-rdc --offload-arch=gfx906 %s -nogpulib -nogpuinc \
// RUN:   2>&1 | FileCheck -check-prefixes=DEV,HOST-PIE \
// RUN:   -DHOST_TRIPLE=x86_64-unknown-linux-gnu %s

// RUN: %clang -### --target=x86_64-unknown-linux-gnu \
// RUN:   --offload-arch=gfx906 %s -nogpulib -nogpuinc \
// RUN:   -fPIC \
// RUN:   2>&1 | FileCheck -check-prefixes=DEV,HOST-PIC \
// RUN:   -DHOST_TRIPLE=x86_64-unknown-linux-gnu %s

// RUN: %clang -### --target=x86_64-unknown-linux-gnu \
// RUN:   -fgpu-rdc --offload-arch=gfx906 %s -nogpulib -nogpuinc \
// RUN:   -fPIC \
// RUN:   2>&1 | FileCheck -check-prefixes=DEV,HOST-PIC \
// RUN:   -DHOST_TRIPLE=x86_64-unknown-linux-gnu %s

// RUN: %clang -### --target=x86_64-unknown-linux-gnu \
// RUN:   --offload-arch=gfx906 %s -nogpulib -nogpuinc \
// RUN:   -fPIE \
// RUN:   2>&1 | FileCheck -check-prefixes=DEV,HOST-PIE \
// RUN:   -DHOST_TRIPLE=x86_64-unknown-linux-gnu %s

// RUN: %clang -### --target=x86_64-unknown-linux-gnu \
// RUN:   -fgpu-rdc --offload-arch=gfx906 %s -nogpulib -nogpuinc \
// RUN:   -fPIE \
// RUN:   2>&1 | FileCheck -check-prefixes=DEV,HOST-PIE \
// RUN:   -DHOST_TRIPLE=x86_64-unknown-linux-gnu %s

// RUN: %clang -### --target=x86_64-unknown-unknown-freebsd \
// RUN:   --offload-arch=gfx906 %s -nogpulib -nogpuinc \
// RUN:   2>&1 | FileCheck -check-prefixes=DEV,HOST-PIE \
// RUN:   -DHOST_TRIPLE=x86_64-unknown-freebsd-unknown %s

// RUN: %clang -### --target=x86_64-unknown-unknown-freebsd \
// RUN:   -fgpu-rdc --offload-arch=gfx906 %s -nogpulib -nogpuinc \
// RUN:   2>&1 | FileCheck -check-prefixes=DEV,HOST-PIE \
// RUN:   -DHOST_TRIPLE=x86_64-unknown-freebsd-unknown %s

// RUN: %clang -### --target=x86_64-unknown-unknown-freebsd \
// RUN:   --offload-arch=gfx906 %s -nogpulib -nogpuinc \
// RUN:   -fPIC \
// RUN:   2>&1 | FileCheck -check-prefixes=DEV,HOST-PIC \
// RUN:   -DHOST_TRIPLE=x86_64-unknown-freebsd-unknown %s

// RUN: %clang -### --target=x86_64-unknown-unknown-freebsd \
// RUN:   -fgpu-rdc --offload-arch=gfx906 %s -nogpulib -nogpuinc \
// RUN:   -fPIC \
// RUN:   2>&1 | FileCheck -check-prefixes=DEV,HOST-PIC \
// RUN:   -DHOST_TRIPLE=x86_64-unknown-freebsd-unknown %s

// RUN: %clang -### --target=x86_64-unknown-unknown-freebsd \
// RUN:   --offload-arch=gfx906 %s -nogpulib -nogpuinc \
// RUN:   -fPIE \
// RUN:   2>&1 | FileCheck -check-prefixes=DEV,HOST-PIE \
// RUN:   -DHOST_TRIPLE=x86_64-unknown-freebsd-unknown %s

// RUN: %clang -### --target=x86_64-unknown-unknown-freebsd \
// RUN:   -fgpu-rdc --offload-arch=gfx906 %s -nogpulib -nogpuinc \
// RUN:   -fPIE \
// RUN:   2>&1 | FileCheck -check-prefixes=DEV,HOST-PIE \
// RUN:   -DHOST_TRIPLE=x86_64-unknown-freebsd-unknown %s

// DEV-DAG: {{".*clang.*".* "-triple" "amdgcn-amd-amdhsa".* "-mrelocation-model" "pic" "-pic-level" "[1|2]".* "-mframe-pointer=all"}}
// HOST-PIC-DAG: {{".*clang.*".* "-triple" "}}[[HOST_TRIPLE]]{{".* "-mrelocation-model" "pic" "-pic-level" "2"}}
// HOST-PIC-NOT: "-pic-is-pie"
// HOST-PIE-DAG: {{".*clang.*".* "-triple" "}}[[HOST_TRIPLE]]{{".* "-mrelocation-model" "pic" "-pic-level" "2" "-pic-is-pie"}}
// DEV-NOT: {{".*clang.*".* "-triple" "amdgcn-amd-amdhsa".* "-pic-is-pie"}}
