// Normalize target features before forwarding them to the linker.
// RUN: %clang -### --target=amdgpu10.1-amd-amdhsa -nostdlib \
// RUN:   -mno-cumode -mcumode %s 2>&1 | FileCheck %s --check-prefix=ON
// RUN: %clang -### --target=amdgpu10.1-amd-amdhsa -nostdlib \
// RUN:   -mcumode -mno-cumode %s 2>&1 | FileCheck %s --check-prefix=OFF
// RUN: %clang -### --target=amdgpu10.1-amd-amdhsa -nostdlib \
// RUN:   -mcumode -mcumode %s 2>&1 | FileCheck %s --check-prefix=ON
// RUN: %clang -### --target=amdgpu10.1-amd-amdhsa -nostdlib \
// RUN:   -mno-cumode -mno-cumode %s 2>&1 | FileCheck %s --check-prefix=OFF

// ON: "-target-feature" "+cumode"
// ON: "-plugin-opt=-mattr=+cumode"
// OFF: "-target-feature" "-cumode"
// OFF: "-plugin-opt=-mattr=-cumode"

// Preserve the relative order of surviving, independently selected features.
// RUN: %clang -### --target=amdgpu10.1-amd-amdhsa -nostdlib \
// RUN:   -mcumode -mtgsplit -mno-cumode %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ORDER
// ORDER: "-plugin-opt=-mattr=+tgsplit,-cumode"

// Do not emit an empty feature argument.
// RUN: %clang -### --target=amdgpu10.1-amd-amdhsa -nostdlib \
// RUN:   %s 2>&1 | FileCheck %s --check-prefix=DEFAULT
// DEFAULT: ld.lld
// DEFAULT-NOT: "-plugin-opt=-mattr=
