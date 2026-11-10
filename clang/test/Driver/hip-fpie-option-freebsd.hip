// -fPIC and -fPIE only affects host relocation model.
// device compilation always uses PIC.

// RUN: %clang -### --target=x86_64-unknown-freebsd \
// RUN:   --offload-arch=gfx906 %s -nogpulib -nogpuinc \
// RUN:   2>&1 | FileCheck -check-prefixes=DEV,HOST-PIE %s

// RUN: %clang -### --target=x86_64-unknown-freebsd \
// RUN:   -fgpu-rdc --offload-arch=gfx906 %s -nogpulib -nogpuinc \
// RUN:   2>&1 | FileCheck -check-prefixes=DEV,HOST-PIE %s

// RUN: %clang -### --target=x86_64-unknown-freebsd \
// RUN:   --offload-arch=gfx906 %s -nogpulib -nogpuinc \
// RUN:   -fPIC \
// RUN:   2>&1 | FileCheck -check-prefixes=DEV,HOST-PIC %s

// RUN: %clang -### --target=x86_64-unknown-freebsd \
// RUN:   -fgpu-rdc --offload-arch=gfx906 %s -nogpulib -nogpuinc \
// RUN:   -fPIC \
// RUN:   2>&1 | FileCheck -check-prefixes=DEV,HOST-PIC %s

// DEV-DAG: {{".*clang.*".* "-triple" "amdgpu9.06-amd-amdhsa".* "-mrelocation-model" "pic" "-pic-level" "[1|2]".* "-mframe-pointer=all"}}
// HOST-PIC-DAG: {{".*clang.*".* "-triple" "x86_64-unknown-freebsd".* "-mrelocation-model" "pic" "-pic-level" "2"}}
// HOST-PIC-NOT: "-pic-is-pie"
// HOST-PIE-DAG: {{".*clang.*".* "-triple" "x86_64-unknown-freebsd".* "-mrelocation-model" "pic" "-pic-level" "2" "-pic-is-pie"}}
// DEV-NOT: {{".*clang.*".* "-triple" "amdgcn-amd-amdhsa".* "-pic-is-pie"}}
