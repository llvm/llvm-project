// REQUIRES: amdgpu-registered-target, default-pie-on-linux

// -fPIC and -fPIE only affects host relocation model.
// device compilation always uses PIC. 

// RUN: %clang -### -target x86_64-unknown-linux-gnu \
// RUN:   --offload-arch=gfx906 %s -nogpulib -nogpuinc \
// RUN:   2>&1 | FileCheck -check-prefixes=DEV,HOST-PIE %s

// RUN: %clang -### -target x86_64-unknown-linux-gnu \
// RUN:   -fgpu-rdc --offload-arch=gfx906 %s -nogpulib -nogpuinc \
// RUN:   2>&1 | FileCheck -check-prefixes=DEV,HOST-PIE %s

// RUN: %clang -### -target x86_64-unknown-linux-gnu \
// RUN:   --offload-arch=gfx906 %s -nogpulib -nogpuinc \
// RUN:   -fPIC \
// RUN:   2>&1 | FileCheck -check-prefixes=DEV,HOST-PIC %s

// RUN: %clang -### -target x86_64-unknown-linux-gnu \
// RUN:   -fgpu-rdc --offload-arch=gfx906 %s -nogpulib -nogpuinc \
// RUN:   -fPIC \
// RUN:   2>&1 | FileCheck -check-prefixes=DEV,HOST-PIC %s

// RUN: %clang -### -target x86_64-unknown-linux-gnu \
// RUN:   --offload-arch=gfx906 %s -nogpulib -nogpuinc \
// RUN:   -fPIE \
// RUN:   2>&1 | FileCheck -check-prefixes=DEV,HOST-PIE %s

// RUN: %clang -### -target x86_64-unknown-linux-gnu \
// RUN:   -fgpu-rdc --offload-arch=gfx906 %s -nogpulib -nogpuinc \
// RUN:   -fPIE \
// RUN:   2>&1 | FileCheck -check-prefixes=DEV,HOST-PIE %s

// DEV-DAG: {{".*clang.*".* "-triple" "amdgcn-amd-amdhsa".* "-mrelocation-model" "pic" "-pic-level" "[1|2]".* "-mframe-pointer=all"}}
// HOST-PIC-DAG: {{".*clang.*".* "-triple" "x86_64-unknown-linux-gnu".* "-mrelocation-model" "pic" "-pic-level" "2"}}
// HOST-PIC-NOT: "-pic-is-pie"
// HOST-PIE-DAG: {{".*clang.*".* "-triple" "x86_64-unknown-linux-gnu".* "-mrelocation-model" "pic" "-pic-level" "2" "-pic-is-pie"}}
// DEV-NOT: {{".*clang.*".* "-triple" "amdgcn-amd-amdhsa".* "-pic-is-pie"}}
