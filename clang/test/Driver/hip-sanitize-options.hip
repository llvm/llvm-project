// REQUIRES: x86-registered-target, amdgpu-registered-target

// RUN: %clang -### -target x86_64-unknown-linux-gnu --offload-arch=gfx900:xnack+ \
// RUN:   -fsanitize=address \
// RUN:   -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefixes=NORDC %s

// RUN: %clang -### -target x86_64-unknown-linux-gnu --offload-arch=gfx900:xnack+ \
// RUN:   -fsanitize=address -fno-gpu-sanitize \
// RUN:   -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck %s

// RUN: %clang -### -target x86_64-unknown-linux-gnu --offload-arch=gfx900:xnack+ \
// RUN:   -fsanitize=address -fgpu-sanitize \
// RUN:   -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefixes=NORDC %s

// RUN: %clang -### -target x86_64-unknown-linux-gnu --offload-arch=gfx900:xnack+ \
// RUN:   -fsanitize=address -fgpu-sanitize -fgpu-rdc \
// RUN:   -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefixes=RDC %s

// RUN: %clang -### -target x86_64-unknown-linux-gnu --offload-arch=gfx900:xnack+ \
// RUN:   -fsanitize=address -fgpu-sanitize \
// RUN:   -nogpuinc --rocm-path=%S/Inputs/rocm-invalid \
// RUN:   %s 2>&1 | FileCheck -check-prefixes=FAIL %s

// RUN: %clang -### -target x86_64-unknown-linux-gnu --offload-arch=gfx900:xnack- \
// RUN:   --offload-arch=gfx900:xnack+ --offload-arch=gfx906 -fsanitize=address -fgpu-sanitize \
// RUN:   -fsanitize=leak -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefixes=XNACK %s

// RUN: %clang -### -target x86_64-unknown-linux-gnu --offload-arch=gfx900:xnack- \
// RUN:   --offload-arch=gfx900:xnack+ --offload-arch=gfx906 -fsanitize=address -fgpu-sanitize \
// RUN:   -fsanitize=leak -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefixes=XNACKNEG %s

// Check -fno-gpu-sanitize with supported and unsupported GPU.
// Sanitizer should be disabled for all GPU's and there should be no warnings about unsupported
// GPU.

// RUN: %clang -### -target x86_64-unknown-linux-gnu --offload-arch=gfx900:xnack- \
// RUN:   --offload-arch=gfx900:xnack+ --offload-arch=gfx906 -fsanitize=address -fno-gpu-sanitize \
// RUN:   -fsanitize=leak -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefixes=NOGPU %s

// RUN: %clang -### -target x86_64-unknown-linux-gnu --offload-arch=gfx900:xnack- \
// RUN:   --offload-arch=gfx900:xnack+ --offload-arch=gfx906 -fsanitize=address -fno-gpu-sanitize \
// RUN:   -fsanitize=leak -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefixes=NOGPUNEG %s

// CHECK-NOT: {{"[^"]*clang[^"]*".* "-fcuda-is-device".* "-fsanitize=address"}}
// CHECK-NOT: {{"[^"]*clang[^"]*".* "-fcuda-is-device".* "-mlink-bitcode-file" ".*asanrtl.bc"}}
// CHECK-NOT: {{"[^"]*lld(\.exe){0,1}".* ".*hip.bc"}}
// CHECK: {{"[^"]*clang[^"]*".* "-triple" "x86_64-unknown-linux-gnu".* "-fsanitize=address"}}

// NORDC: {{"[^"]*clang[^"]*".* "-emit-obj".* "-fcuda-is-device".* "-mlink-bitcode-file" ".*asanrtl.bc".* "-mlink-builtin-bitcode" ".*hip.bc".* "-fsanitize=address".*}} "-o" "[[OUT:[^"]*.o]]"
// NORDC-NOT: {{"[^"]*lld(\.exe){0,1}".*}} "[[OUT]]" {{".*asanrtl.bc" ".*hip.bc"}}
// NORDC: {{"[^"]*clang[^"]*".* "-triple" "x86_64-unknown-linux-gnu".* "-fsanitize=address"}}

// RDC: {{"[^"]*clang[^"]*".* "-triple" "x86_64-unknown-linux-gnu".* "-fsanitize=address"}}
// RDC: {{"[^"]*clang[^"]*".* "-emit-llvm-bc".* "-fcuda-is-device".* "-mlink-bitcode-file" ".*asanrtl.bc".* "-mlink-builtin-bitcode" ".*hip.bc".* "-fsanitize=address".*}} "-o" "[[OUT:[^"]*.bc]]"
// RDC-NOT: {{"[^"]*lld(\.exe){0,1}".*}} "[[OUT]]" {{".*asanrtl.bc" ".*hip.bc"}}

// FAIL: AMDGPU address sanitizer runtime library (asanrtl) is not found. Please install ROCm device library which supports address sanitizer

// XNACK-DAG: warning: ignoring '-fsanitize=leak' option as it is not currently supported for target 'amdgcn-amd-amdhsa'
// XNACK-DAG: warning: ignoring '-fsanitize=address' option as it is not currently supported for offload arch 'gfx900:xnack-'. Use it with an offload arch containing 'xnack+' instead
// XNACK-DAG: warning: ignoring '-fsanitize=address' option as it is not currently supported for offload arch 'gfx906'. Use it with an offload arch containing 'xnack+' instead
// XNACK-DAG: {{"[^"]*clang[^"]*".* "-mlink-bitcode-file" ".*asanrtl.bc".* "-target-cpu" "gfx900".* "-target-feature" "\+xnack".* "-fsanitize=address"}}
// XNACK-DAG: {{"[^"]*clang[^"]*".* "-target-cpu" "gfx900".* "-target-feature" "-xnack"}}
// XNACK-DAG: {{"[^"]*clang[^"]*".* "-target-cpu" "gfx906"}}
// XNACK-DAG: {{"[^"]*clang[^"]*".* "-triple" "x86_64-unknown-linux-gnu".* "-fsanitize=address,leak"}}
// XNACKNEG-NOT: {{"[^"]*clang[^"]*".* "-target-cpu" "gfx900".* "-target-feature" "+xnack".* "-fsanitize=address,leak"}}
// XNACKNEG-NOT: {{"[^"]*clang[^"]*".* "-target-cpu" "gfx900".* "-target-feature" "-xnack".* "-fsanitize=address,leak"}}
// XNACKNEG-NOT: {{"[^"]*clang[^"]*".* "-target-cpu" "gfx906".* "-fsanitize=address,leak"}}
// XNACKNEG-NOT: {{"[^"]*clang[^"]*".* "-target-cpu" "gfx900".* "-target-feature" "-xnack".* "-fsanitize=address"}}
// XNACKNEG-NOT: {{"[^"]*clang[^"]*".* "-target-cpu" "gfx906".* "-fsanitize=address"}}
// XNACKNEG-NOT: {{"[^"]*clang[^"]*".* "-mlink-bitcode-file" ".*asanrtl.bc".* "-target-cpu" "gfx900".* "-target-feature" "-xnack"}}
// XNACKNEG-NOT: {{"[^"]*clang[^"]*".* "-mlink-bitcode-file" ".*asanrtl.bc".* "-target-cpu" "gfx906"}}
// XNACKNEG-NOT: {{"[^"]*lld(\.exe){0,1}".* ".*hip.bc"}}

// NOGPU-DAG: {{"[^"]*clang[^"]*".* "-target-cpu" "gfx900".* "-target-feature" "\+xnack"}}
// NOGPU-DAG: {{"[^"]*clang[^"]*".* "-target-cpu" "gfx900".* "-target-feature" "-xnack"}}
// NOGPU-DAG: {{"[^"]*clang[^"]*".* "-target-cpu" "gfx906"}}
// NOGPU-DAG: {{"[^"]*clang[^"]*".* "-triple" "x86_64-unknown-linux-gnu".* "-fsanitize=address,leak"}}
// NOGPUNEG-NOT: warning: ignoring '-fsanitize=leak' option as it is not currently supported for target 'amdgcn-amd-amdhsa'
// NOGPUNEG-NOT: warning: ignoring '-fsanitize=address' option as it is not currently supported for offload arch 'gfx900:xnack-'. Use it with an offload arch containing 'xnack+' instead
// NOGPUNEG-NOT: warning: ignoring '-fsanitize=address' option as it is not currently supported for offload arch 'gfx906'. Use it with an offload arch containing 'xnack+' instead
// NOGPUNEG-NOT: {{"[^"]*clang[^"]*".* "-mlink-bitcode-file" ".*asanrtl.bc".* "-target-cpu" "gfx900".* "-target-feature" "\+xnack".* "-fsanitize=address"}}
// NOGPUNEG-NOT: {{"[^"]*clang[^"]*".* "-target-cpu" "gfx900".* "-target-feature" "\+xnack".* "-fsanitize=address,leak"}}
// NOGPUNEG-NOT: {{"[^"]*clang[^"]*".* "-target-cpu" "gfx900".* "-target-feature" "-xnack".* "-fsanitize=address,leak"}}
// NOGPUNEG-NOT: {{"[^"]*clang[^"]*".* "-target-cpu" "gfx906".* "-fsanitize=address,leak"}}
// NOGPUNEG-NOT: {{"[^"]*clang[^"]*".* "-target-cpu" "gfx900".* "-target-feature" "\+xnack".* "-fsanitize=address"}}
// NOGPUNEG-NOT: {{"[^"]*clang[^"]*".* "-target-cpu" "gfx900".* "-target-feature" "-xnack".* "-fsanitize=address"}}
// NOGPUNEG-NOT: {{"[^"]*clang[^"]*".* "-target-cpu" "gfx906".* "-fsanitize=address"}}
// NOGPUNEG-NOT: {{"[^"]*clang[^"]*".* "-mlink-bitcode-file" ".*asanrtl.bc".* "-target-cpu" "gfx900".* "-target-feature" "\+xnack"}}
// NOGPUNEG-NOT: {{"[^"]*clang[^"]*".* "-mlink-bitcode-file" ".*asanrtl.bc".* "-target-cpu" "gfx900".* "-target-feature" "-xnack"}}
// NOGPUNEG-NOT: {{"[^"]*clang[^"]*".* "-mlink-bitcode-file" ".*asanrtl.bc".* "-target-cpu" "gfx906"}}
// NOGPUNEG-NOT: {{"[^"]*lld(\.exe){0,1}".* ".*hip.bc"}}
