// REQUIRES: clang-driver, amdgpu-registered-target

// Check bundle ID for code object v2.

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   -mno-code-object-v3 \
// RUN:   --offload-arch=gfx906 -nogpulib \
// RUN:   %s 2>&1 | FileCheck -check-prefixes=V2,V2-WARN %s

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   -mcode-object-version=2 \
// RUN:   --offload-arch=gfx906 -nogpulib \
// RUN:   %s 2>&1 | FileCheck -check-prefix=V2 %s

// V2-WARN: warning: argument '-mno-code-object-v3' is deprecated, use '-mcode-object-version=2' instead [-Wdeprecated]
// V2: "-mllvm" "--amdhsa-code-object-version=2"
// V2: "-targets=host-x86_64-unknown-linux,hip-amdgcn-amd-amdhsa--gfx906"

// Check bundle ID for code object v3.

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   -mcode-object-v3 \
// RUN:   --offload-arch=gfx906 -nogpulib \
// RUN:   %s 2>&1 | FileCheck -check-prefixes=V3,V3-WARN %s

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   -mcode-object-version=3 \
// RUN:   --offload-arch=gfx906 -nogpulib \
// RUN:   %s 2>&1 | FileCheck -check-prefix=V3 %s

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   -mcode-object-version=4 -mcode-object-version=3 \
// RUN:   --offload-arch=gfx906 -nogpulib \
// RUN:   %s 2>&1 | FileCheck -check-prefix=V3 %s

// V3-WARN: warning: argument '-mcode-object-v3' is deprecated, use '-mcode-object-version=3' instead [-Wdeprecated]
// V3: "-mllvm" "--amdhsa-code-object-version=3"
// V3: "-targets=host-x86_64-unknown-linux,hip-amdgcn-amd-amdhsa--gfx906"

// Check bundle ID for code object version 4.

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   -mcode-object-version=4 \
// RUN:   --offload-arch=gfx906 -nogpulib \
// RUN:   %s 2>&1 | FileCheck -check-prefix=V4 %s

// V4: "-mllvm" "--amdhsa-code-object-version=4"
// V4: "-targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx906"

// Check bundle ID for code object version 5.

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   -mcode-object-version=5 \
// RUN:   --offload-arch=gfx906 -nogpulib \
// RUN:   %s 2>&1 | FileCheck -check-prefix=V5 %s

// V5: "-mllvm" "--amdhsa-code-object-version=5"
// V5: "-targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx906"

// Check bundle ID for code object version default

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --offload-arch=gfx906 -nogpulib \
// RUN:   %s 2>&1 | FileCheck -check-prefix=VD %s

// VD: "-targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx906"

// Check invalid code object version option.

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   -mcode-object-version=1 \
// RUN:   --offload-arch=gfx906 -nogpulib \
// RUN:   %s 2>&1 | FileCheck -check-prefix=INVALID %s
// INVALID: error: invalid integral value '1' in '-mcode-object-version=1'
// INVALID-NOT: error: invalid integral value

// Check warnings are emitted for legacy options before -mcode-object-version options.
// Check warnings are emitted only once.

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   -mno-code-object-v3 -mcode-object-v3 -mcode-object-version=4 \
// RUN:   --offload-arch=gfx906 -nogpulib \
// RUN:   %s 2>&1 | FileCheck -check-prefixes=WARN %s
// WARN: warning: argument '-mno-code-object-v3' is deprecated, use '-mcode-object-version=2' instead [-Wdeprecated]
// WARN: warning: argument '-mcode-object-v3' is deprecated, use '-mcode-object-version=3' instead [-Wdeprecated]
// WARN-NOT: warning: argument {{.*}} is deprecated
