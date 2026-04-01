// RUN: %clang -### --target=x86_64-unknown-linux-gnu --offload-arch=gfx900:xnack+ \
// RUN:   -fsanitize=address \
// RUN:   -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefixes=NORDC %s

// RUN: %clang -### --target=x86_64-unknown-linux-gnu --offload-arch=gfx1250,gfx1251 \
// RUN:   -fsanitize=address \
// RUN:   -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefixes=NORDC %s

// RUN: %clang -### --target=x86_64-unknown-linux-gnu --offload-arch=gfx900:xnack+ \
// RUN:   -fsanitize=address -fno-gpu-sanitize \
// RUN:   -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck %s

// RUN: %clang -### --target=x86_64-unknown-linux-gnu --offload-arch=gfx900:xnack+ \
// RUN:   -fsanitize=address -fgpu-sanitize \
// RUN:   -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefixes=NORDC %s

// RUN: %clang -### --target=x86_64-unknown-linux-gnu --offload-arch=gfx900:xnack+ \
// RUN:   -fsanitize=address -fgpu-sanitize -fgpu-rdc \
// RUN:   -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefixes=RDC %s

// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -mcode-object-version=5 --offload-arch=gfx900:xnack+ \
// RUN:   -fsanitize=address -fgpu-sanitize \
// RUN:   -nogpuinc --rocm-path=%S/Inputs/rocm-invalid \
// RUN:   %s 2>&1 | FileCheck -check-prefixes=FAIL %s

// RUN: not %clang -### --target=x86_64-unknown-linux-gnu --offload-arch=gfx900:xnack- \
// RUN:   --offload-arch=gfx900:xnack+ --offload-arch=gfx906 -fsanitize=address -fgpu-sanitize \
// RUN:   -fsanitize=leak -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefixes=XNACK %s

// RUN: not %clang -### --target=x86_64-unknown-linux-gnu --offload-arch=gfx900:xnack- \
// RUN:   --offload-arch=gfx900:xnack+ --offload-arch=gfx906 -fsanitize=address -fgpu-sanitize \
// RUN:   -fsanitize=leak -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefixes=XNACKNEG %s

// Check -fno-gpu-sanitize with supported and unsupported GPU.
// Sanitizer should be disabled for all GPU's and there should be no warnings about unsupported
// GPU.

// RUN: not %clang -### --target=x86_64-unknown-linux-gnu --offload-arch=gfx900:xnack- \
// RUN:   --offload-arch=gfx900:xnack+ --offload-arch=gfx906 -fsanitize=address -fno-gpu-sanitize \
// RUN:   -fsanitize=leak -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefixes=NOGPU %s

// RUN: not %clang -### --target=x86_64-unknown-linux-gnu --offload-arch=gfx900:xnack- \
// RUN:   --offload-arch=gfx900:xnack+ --offload-arch=gfx906 -fsanitize=address -fno-gpu-sanitize \
// RUN:   -fsanitize=leak -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefixes=NOGPUNEG %s

// Catch invalid combination of sanitizers regardless of their order and ignore
// them selectively.
// (The address sanitizer enables the device sanitizer pipeline. The fuzzer
// implicitly turns on LLVMs SanitizerCoverage, which the driver then forwards
// to the device cc1. SanitizerCoverage is not supported on amdgcn.)

// RUN: %clang -### --target=x86_64-unknown-linux-gnu --offload-arch=gfx900:xnack+ \
// RUN:   -fsanitize=address,fuzzer --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=INVALIDCOMBINATION,INVALIDCOMBINATION1 %s
// RUN: %clang -### --target=x86_64-unknown-linux-gnu --offload-arch=gfx900:xnack+ \
// RUN:   -fsanitize=fuzzer,address --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=INVALIDCOMBINATION,INVALIDCOMBINATION2 %s

// Do the same for multiple -fsanitize arguments and multi-arch scenarios.

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu --offload-arch=gfx900:xnack+ --offload-arch=gfx908:xnack- \
// RUN:   -fsanitize=address,fuzzer -fsanitize=leak --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=MULT1,XNACK2 %s
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu --offload-arch=gfx900:xnack+,gfx908:xnack- \
// RUN:   -fsanitize=fuzzer,address -fsanitize=leak --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=MULT2,XNACK2 %s

// Check for -fsanitize-coverage options
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu --offload-arch=gfx900:xnack+ \
// RUN:   -fsanitize=address -fsanitize-coverage=inline-bool-flag --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=WARNSANCOV %s

// Test -Xarch_device error scenario

// RUN:   not %clang -### --target=x86_64-unknown-linux-gnu --offload-arch=gfx900:xnack+ \
// RUN:   -Xarch_device -fsanitize=leak --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=UNSUPPORTEDERROR %s

// RUN:   not %clang -### --target=x86_64-unknown-linux-gnu --offload-arch=gfx900:xnack- \
// RUN:   -Xarch_device -fsanitize=address --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=XNACKERROR %s

// RUN:   not %clang -### --target=x86_64-unknown-linux-gnu --offload-arch=gfx900:xnack+ \
// RUN:   -Xarch_device -fsanitize=fuzzer,address --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=INVALIDCOMBINATIONERROR %s

// RUN:   not %clang -### --target=x86_64-unknown-linux-gnu --offload-arch=gfx900:xnack+ \
// RUN:   -fsanitize=address -Xarch_device -fsanitize-coverage-stack-depth-callback-min=42 --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=ERRSANCOV %s

// CHECK-NOT: {{"[^"]*clang[^"]*".* "-fcuda-is-device".* "-fsanitize=address"}}
// CHECK-NOT: {{"[^"]*clang[^"]*".* "-fcuda-is-device".* "-mlink-bitcode-file" ".*asanrtl.bc"}}
// CHECK-NOT: {{"[^"]*lld(\.exe){0,1}".* ".*hip.bc"}}
// CHECK: {{"[^"]*clang[^"]*".* "-triple" "x86_64-unknown-linux-gnu".* "-fsanitize=address"}}

// NORDC: {{"[^"]*clang[^"]*".* "-emit-obj".* "-fcuda-is-device".* .* "-mlink-bitcode-file" ".*asanrtl.bc".* "-fsanitize=address".*}} "-o" "[[OUT:[^"]*.o]]"
// NORDC-NOT: {{"[^"]*lld(\.exe){0,1}".*}} "[[OUT]]" {{".*asanrtl.bc" ".*hip.bc"}}
// NORDC: {{"[^"]*clang[^"]*".* "-triple" "x86_64-unknown-linux-gnu".* "-fsanitize=address"}}

// RDC: {{"[^"]*clang[^"]*".* "-triple" "x86_64-unknown-linux-gnu".* "-fsanitize=address"}}
// RDC: {{"[^"]*clang[^"]*".* "-emit-llvm-bc".* "-fcuda-is-device".* "-mlink-bitcode-file" ".*asanrtl.bc".* "-fsanitize=address".*}} "-o" "[[OUT:[^"]*.bc]]"
// RDC-NOT: {{"[^"]*lld(\.exe){0,1}".*}} "[[OUT]]" {{".*asanrtl.bc" ".*hip.bc"}}

// FAIL: error: cannot find ROCm device library for ABI version 5; provide its path via '--rocm-path' or '--rocm-device-lib-path', or pass '-nogpulib' to build without ROCm device library

// XNACK-DAG: warning: ignoring '-fsanitize=leak' option as it is not currently supported for target 'amdgcn-amd-amdhsa'
// XNACK-DAG: warning: ignoring '-fsanitize=address' option for offload arch 'gfx900:xnack-' as it is not currently supported there. Use it with an offload arch containing 'xnack+' instead
// XNACK-DAG: warning: ignoring '-fsanitize=address' option for offload arch 'gfx906' as it is not currently supported there. Use it with an offload arch containing 'xnack+' instead
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
// NOGPUNEG-NOT: warning: ignoring '-fsanitize=address' option for offload arch 'gfx900:xnack-' as it is not currently supported there. Use it with an offload arch containing 'xnack+' instead
// NOGPUNEG-NOT: warning: ignoring '-fsanitize=address' option for offload arch 'gfx906' as it is not currently supported there. Use it with an offload arch containing 'xnack+' instead
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

// INVALIDCOMBINATION1-DAG: warning: ignoring 'fuzzer' in '-fsanitize=address,fuzzer' option as it is not currently supported for target 'amdgcn-amd-amdhsa' [-Woption-ignored]
// INVALIDCOMBINATION2-DAG: warning: ignoring 'fuzzer' in '-fsanitize=fuzzer,address' option as it is not currently supported for target 'amdgcn-amd-amdhsa' [-Woption-ignored]
// INVALIDCOMBINATION-DAG: {{"[^"]*clang[^"]*".* "-mlink-bitcode-file" ".*asanrtl.bc".* "-target-cpu" "gfx900".* "-target-feature" "\+xnack".* "-fsanitize=address"}}
// INVALIDCOMBINATION-DAG: {{"[^"]*clang[^"]*".* "-triple" "x86_64-unknown-linux-gnu".* "-fsanitize=address,fuzzer,fuzzer-no-link"}}

// MULT1-DAG: warning: ignoring 'fuzzer' in '-fsanitize=address,fuzzer' option as it is not currently supported for target 'amdgcn-amd-amdhsa' [-Woption-ignored]
// MULT1-DAG: warning: ignoring '-fsanitize=leak' option as it is not currently supported for target 'amdgcn-amd-amdhsa' [-Woption-ignored]
// MULT1-DAG: warning: ignoring 'fuzzer' in '-fsanitize=address,fuzzer' option as it is not currently supported for target 'amdgcn-amd-amdhsa' [-Woption-ignored]
// MULT1-DAG: warning: ignoring '-fsanitize=address,fuzzer' option for offload arch 'gfx908:xnack-' as it is not currently supported there. Use it with an offload arch containing 'xnack+' instead [-Woption-ignored]
// MULT1-DAG: warning: ignoring '-fsanitize=leak' option as it is not currently supported for target 'amdgcn-amd-amdhsa' [-Woption-ignored]

// MULT2-DAG: warning: ignoring 'fuzzer' in '-fsanitize=fuzzer,address' option as it is not currently supported for target 'amdgcn-amd-amdhsa' [-Woption-ignored]
// MULT2-DAG: warning: ignoring '-fsanitize=leak' option as it is not currently supported for target 'amdgcn-amd-amdhsa' [-Woption-ignored]
// MULT2-DAG: warning: ignoring 'fuzzer' in '-fsanitize=fuzzer,address' option as it is not currently supported for target 'amdgcn-amd-amdhsa' [-Woption-ignored]
// MULT2-DAG: warning: ignoring '-fsanitize=fuzzer,address' option for offload arch 'gfx908:xnack-' as it is not currently supported there. Use it with an offload arch containing 'xnack+' instead [-Woption-ignored]
// MULT2-DAG: warning: ignoring '-fsanitize=leak' option as it is not currently supported for target 'amdgcn-amd-amdhsa' [-Woption-ignored]

// XNACK2-DAG: {{"[^"]*clang[^"]*".* "-mlink-bitcode-file" ".*asanrtl.bc".* "-target-cpu" "gfx900".* "-target-feature" "\+xnack".* "-fsanitize=address"}}
// XNACK2-DAG: {{"[^"]*clang[^"]*".* "-target-cpu" "gfx908"}}
// XNACK2-DAG: {{"[^"]*clang[^"]*".* "-triple" "x86_64-unknown-linux-gnu".* "-fsanitize=address,fuzzer,fuzzer-no-link,leak"}}

// UNSUPPORTEDERROR: error: '-fsanitize=leak' option is not currently supported for target 'amdgcn-amd-amdhsa'
// XNACKERROR: error: '-fsanitize=address' option for offload arch 'gfx900:xnack-' is not currently supported there. Use it with an offload arch containing 'xnack+' instead
// INVALIDCOMBINATIONERROR: error: 'fuzzer' in '-fsanitize=fuzzer,address' option is not currently supported for target 'amdgcn-amd-amdhsa'

// WARNSANCOV: warning: ignoring '-fsanitize-coverage=inline-bool-flag' option as it is not currently supported for target 'amdgcn-amd-amdhsa'
// ERRSANCOV: error: '-fsanitize-coverage-stack-depth-callback-min=42' option is not currently supported for target 'amdgcn-amd-amdhsa'
