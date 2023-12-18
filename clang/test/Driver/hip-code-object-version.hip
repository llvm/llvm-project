// REQUIRES: amdgpu-registered-target


// Check bundle ID for code object version 4.

// RUN: not %clang -### --target=x86_64-linux-gnu \
// RUN:   -mcode-object-version=4 \
// RUN:   --offload-arch=gfx906 --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefix=V4 %s

// V4: "-mcode-object-version=4"
// V4: "-mllvm" "--amdhsa-code-object-version=4"
// V4: "-targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx906"

// Check bundle ID for code object version 5.

// RUN: not %clang -### --target=x86_64-linux-gnu \
// RUN:   -mcode-object-version=5 \
// RUN:   --offload-arch=gfx906 --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefix=V5 %s

// V5: "-mcode-object-version=5"
// V5: "-mllvm" "--amdhsa-code-object-version=5"
// V5: "-targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx906"

// Check bundle ID for code object version default

// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --offload-arch=gfx906 -nogpuinc -nogpulib \
// RUN:   %s 2>&1 | FileCheck -check-prefix=VD %s

// VD: "-targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx906"

// Check invalid code object version option.

// RUN: not %clang -### --target=x86_64-linux-gnu \
// RUN:   -mcode-object-version=1 \
// RUN:   --offload-arch=gfx906 --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefix=INVALID_1 %s
// INVALID_1: error: invalid integral value '1' in '-mcode-object-version=1'
// INVALID_1-NOT: error: invalid integral value

// RUN: not %clang -### --target=x86_64-linux-gnu \
// RUN:   -mcode-object-version=2 \
// RUN:   --offload-arch=gfx906 --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefix=INVALID_2 %s
// INVALID_2: error: invalid integral value '2' in '-mcode-object-version=2'
// INVALID_2-NOT: error: invalid integral value

// RUN: not %clang -### --target=x86_64-linux-gnu \
// RUN:   -mcode-object-version=3 \
// RUN:   --offload-arch=gfx906 --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefix=INVALID_3 %s
// INVALID_3: error: invalid integral value '3' in '-mcode-object-version=3'
// INVALID_3-NOT: error: invalid integral value

// Check LLVM code object version option --amdhsa-code-object-version
// is passed to -cc1 and -cc1as, and -mcode-object-version is passed
// to -cc1 but not -cc1as.

// RUN: not %clang -### --target=x86_64-linux-gnu \
// RUN:   -mcode-object-version=5 \
// RUN:   --offload-arch=gfx906 --rocm-path=%S/Inputs/rocm -save-temps \
// RUN:   %s 2>&1 | FileCheck -check-prefix=CC1 %s

// CC1: "-cc1" {{.*}}"-mcode-object-version=5" {{.*}}"-mllvm" "--amdhsa-code-object-version=5"
// CC1: "-cc1as" {{.*}}"-mllvm" "--amdhsa-code-object-version=5"

// RUN: not %clang -### --target=x86_64-linux-gnu \
// RUN:   -mcode-object-version=5 \
// RUN:   --offload-arch=gfx906 --rocm-path=%S/Inputs/rocm -save-temps \
// RUN:   %s 2>&1 | FileCheck -check-prefix=CC1NEG %s

// CC1NEG-NOT: "-cc1as" {{.*}}"-mcode-object-version=5"
