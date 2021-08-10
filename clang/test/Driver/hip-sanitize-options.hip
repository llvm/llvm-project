// REQUIRES: clang-driver, x86-registered-target, amdgpu-registered-target

// RUN: %clang -### -target x86_64-unknown-linux-gnu --offload-arch=gfx900 \
// RUN:   -fsanitize=address \
// RUN:   -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck %s

// RUN: %clang -### -target x86_64-unknown-linux-gnu --offload-arch=gfx900 \
// RUN:   -fsanitize=address -fno-gpu-sanitize \
// RUN:   -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck %s

// RUN: %clang -### -target x86_64-unknown-linux-gnu --offload-arch=gfx900 \
// RUN:   -fsanitize=address -fgpu-sanitize \
// RUN:   -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefixes=NORDC %s

// RUN: %clang -### -target x86_64-unknown-linux-gnu --offload-arch=gfx900 \
// RUN:   -fsanitize=address -fgpu-sanitize -fgpu-rdc \
// RUN:   -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefixes=RDC %s

// RUN: %clang -### -target x86_64-unknown-linux-gnu --offload-arch=gfx900 \
// RUN:   -fsanitize=address -fgpu-sanitize \
// RUN:   -nogpuinc --rocm-path=%S/Inputs/rocm-invalid \
// RUN:   %s 2>&1 | FileCheck -check-prefixes=FAIL %s

// RUN: %clang -### -target x86_64-unknown-linux-gnu --offload-arch=gfx900:xnack- \
// RUN:   -fsanitize=address -fgpu-sanitize \
// RUN:   -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefix=XNACK %s

// CHECK-NOT: {{"[^"]*clang[^"]*".* "-fcuda-is-device".* "-fsanitize=address"}}
// CHECK-NOT: {{"[^"]*lld(\.exe){0,1}".* ".*hip.bc"}}
// CHECK: {{"[^"]*clang[^"]*".* "-triple" "x86_64-unknown-linux-gnu".* "-fsanitize=address"}}

// NORDC: {{"[^"]*clang[^"]*".* "-emit-obj".* "-fcuda-is-device".* "-fsanitize=address".*}} "-o" "[[OUT:[^"]*.o]]"
// NORDC: {{"[^"]*lld(\.exe){0,1}".*}} "[[OUT]]" {{".*asanrtl.bc" ".*hip.bc"}}
// NORDC: {{"[^"]*clang[^"]*".* "-triple" "x86_64-unknown-linux-gnu".* "-fsanitize=address"}}

// RDC: {{"[^"]*clang[^"]*".* "-triple" "x86_64-unknown-linux-gnu".* "-fsanitize=address"}}
// RDC: {{"[^"]*clang[^"]*".* "-emit-llvm-bc".* "-fcuda-is-device".* "-fsanitize=address".*}} "-o" "[[OUT:[^"]*.bc]]"
// RDC: {{"[^"]*lld(\.exe){0,1}".*}} "[[OUT]]" {{".*asanrtl.bc" ".*hip.bc"}}

// FAIL: AMDGPU address sanitizer runtime library (asanrtl) is not found. Please install ROCm device library which supports address sanitizer

// XNACK: error: '-fgpu-sanitize' is not compatible with offload arch 'gfx900:xnack-'. Use an offload arch without 'xnack-' instead
