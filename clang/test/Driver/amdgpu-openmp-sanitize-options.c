// REQUIRES: x86-registered-target, amdgpu-registered-target

// Fail on invalid ROCm Path.
// RUN:   not %clang -no-canonical-prefixes -### -mcode-object-version=5 --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=gfx908:xnack+ -fsanitize=address -fgpu-sanitize -nogpuinc --rocm-path=%S/Inputs/rocm-invalid  %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FAIL %s

// Enable multiple sanitizer's apart from ASan with invalid rocm-path.
// RUN:   not %clang -no-canonical-prefixes -### -mcode-object-version=5 --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=gfx908:xnack+ -fsanitize=address -fsanitize=leak -fgpu-sanitize --rocm-path=%S/Inputs/rocm-invalid -nogpuinc  %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NOTSUPPORTED,FAIL %s

// Memory, Leak, UndefinedBehaviour and Thread Sanitizer are not supported on AMDGPU.
// RUN:   %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=gfx908:xnack+ -fsanitize=address -fsanitize=leak -fgpu-sanitize --rocm-path=%S/Inputs/rocm -nogpuinc  %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NOTSUPPORTED %s

// GPU ASan Enabled Test Cases

// GPU ASan enabled for amdgpu-arch [gfx908:xnack+]
// RUN:   %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=gfx908:xnack+ -fsanitize=address -fgpu-sanitize --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=HOSTSAN,GPUSAN,SAN %s

// GPU ASan enabled through '-fsanitize=address' flag  without '-fgpu-sanitize' for amdgpu-arch [gfx908:xnack+]
// RUN:   %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=gfx908:xnack+ -fsanitize=address --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=HOSTSAN,GPUSAN,SAN %s

// GPU ASan enabled for multiple amdgpu-arch [gfx908:xnack+,gfx900:xnack+]
// RUN:   %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=gfx908:xnack+ --offload-arch=gfx900:xnack+ -fsanitize=address -fgpu-sanitize --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=HOSTSAN,GPUSAN,SAN %s

// GPU ASan enabled  for amdgpu-arch [gfx1250,gfx1251]
// RUN:   %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=gfx1250,gfx1251 -fsanitize=address -fgpu-sanitize --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=HOSTSAN,GPUSAN,SAN %s

// GPU ASan Disabled Test Cases

// GPU ASan disabled through '-fsanitize=address' without '-fgpu-sanitize' flag for amdgpu-arch [gfx908]
// RUN:   %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=gfx908 -fsanitize=address --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=NOXNACK,HOSTSAN,NOGPUSAN,SAN %s

// GPU ASan disabled for amdgpu-arch [gfx908]
// RUN:   %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=gfx908 -fsanitize=address -fgpu-sanitize --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=NOXNACK,HOSTSAN,NOGPUSAN,SAN %s

// GPU ASan disabled for amdgpu-arch [gfx908:xnack-]
// RUN:   %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=gfx908:xnack- -fsanitize=address -fgpu-sanitize --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=XNACKNEG,HOSTSAN,NOGPUSAN,SAN %s

// GPU ASan disabled using '-fno-gpu-sanitize' for amdgpu-arch [gfx908:xnack+]
// RUN:   %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=gfx908:xnack+ -fsanitize=address -fno-gpu-sanitize --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=HOSTSAN,NOGPUSAN,SAN %s

// GPU ASan disabled for multiple amdgpu-arch [gfx908:xnack+,gfx900:xnack+]
// RUN:   %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=gfx908:xnack+ --offload-arch=gfx900:xnack+ -fsanitize=address -fno-gpu-sanitize --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=HOSTSAN,NOGPUSAN,SAN %s

// Catch invalid combination of sanitizers regardless of their order and ignore
// them selectively.
// (The address sanitizer enables the device sanitizer pipeline. The fuzzer
// implicitly turns on LLVMs SanitizerCoverage, which the driver then forwards
// to the device cc1. SanitizerCoverage is not supported on amdgcn.)

// RUN:   %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=gfx908:xnack+ -fsanitize=address,fuzzer --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=HOSTSANCOMBINATION,INVALIDCOMBINATION1 %s
// RUN:   %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=gfx908:xnack+ -fsanitize=fuzzer,address --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=HOSTSANCOMBINATION,INVALIDCOMBINATION2 %s

// Do the same for multiple -fsanitize arguments and multi-arch scenarios.

// RUN:   %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=gfx908:xnack+ --offload-arch=gfx900:xnack- -fsanitize=address,fuzzer --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=HOSTSANCOMBINATION,INVALIDCOMBINATION1 %s
// RUN:   %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=gfx908:xnack+,gfx900:xnack- -fsanitize=address,fuzzer --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=HOSTSANCOMBINATION,INVALIDCOMBINATION1 %s
// RUN:   %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=gfx908:xnack+,gfx900:xnack- -fsanitize=fuzzer,address -fsanitize=leak --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=HOSTSANCOMBINATION2,NOTSUPPORTED-DAG,INVALIDCOMBINATION2 %s

// Check for -fsanitize-coverage options
// RUN:   %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=gfx908:xnack+ -fsanitize=address -fsanitize-coverage=inline-bool-flag --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=WARNSANCOV %s

// Test -Xarch_device error scenario

// RUN:   not %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=gfx908:xnack+ -Xarch_device -fsanitize=leak --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=UNSUPPORTEDERROR %s

// RUN:   not %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=gfx908:xnack- -Xarch_device -fsanitize=address --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=XNACKERROR %s

// RUN:   not %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=gfx908:xnack+ -Xarch_device -fsanitize=fuzzer,address --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=INVALIDCOMBINATIONERROR %s

// RUN:   not %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=gfx908:xnack+ -fsanitize=address -Xarch_device -fsanitize-coverage-stack-depth-callback-min=42 --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=ERRSANCOV %s


// INVALIDCOMBINATION1: warning: ignoring 'fuzzer' in '-fsanitize=address,fuzzer' option as it is not currently supported for target 'amdgcn-amd-amdhsa' [-Woption-ignored]
// INVALIDCOMBINATION2: warning: ignoring 'fuzzer' in '-fsanitize=fuzzer,address' option as it is not currently supported for target 'amdgcn-amd-amdhsa' [-Woption-ignored]

// FAIL-DAG: error: cannot find ROCm device library for ABI version 5; provide its path via '--rocm-path' or '--rocm-device-lib-path', or pass '-nogpulib' to build without ROCm device library
// NOTSUPPORTED-DAG: warning: ignoring '-fsanitize=leak' option as it is not currently supported for target 'amdgcn-amd-amdhsa'

// NOXNACK: warning: ignoring '-fsanitize=address' option for offload arch 'gfx908' as it is not currently supported there. Use it with an offload arch containing 'xnack+' instead
// XNACKNEG: warning: ignoring '-fsanitize=address' option for offload arch 'gfx908:xnack-' as it is not currently supported there. Use it with an offload arch containing 'xnack+' instead

// HOSTSAN: {{"[^"]*clang[^"]*" "-cc1" "-triple" "x86_64-unknown-linux-gnu".* "-fopenmp".* "-fsanitize=address".* "--offload-targets=amdgcn-amd-amdhsa".* "-x" "c".*}}
// HOSTSANCOMBINATION: {{"[^"]*clang[^"]*" "-cc1" "-triple" "x86_64-unknown-linux-gnu".* "-fopenmp".* "-fsanitize=address,fuzzer,fuzzer-no-link".* "--offload-targets=amdgcn-amd-amdhsa".* "-x" "c".*}}
// HOSTSANCOMBINATION2: {{"[^"]*clang[^"]*" "-cc1" "-triple" "x86_64-unknown-linux-gnu".* "-fopenmp".* "-fsanitize=address,fuzzer,fuzzer-no-link,leak".* "--offload-targets=amdgcn-amd-amdhsa".* "-x" "c".*}}

// GPUSAN: {{"[^"]*clang[^"]*" "-cc1" "-triple" "amdgcn-amd-amdhsa" "-aux-triple" "x86_64-unknown-linux-gnu".* "-emit-llvm-bc".* "-mlink-bitcode-file" "[^"]*asanrtl.bc".* "-mlink-bitcode-file" "[^"]*ockl.bc".* "-target-cpu" "(gfx908|gfx900|gfx1250|gfx1251)".* "-fopenmp".* "-fsanitize=address".* "-x" "c".*}}
// NOGPUSAN: {{"[^"]*clang[^"]*" "-cc1" "-triple" "amdgcn-amd-amdhsa" "-aux-triple" "x86_64-unknown-linux-gnu".* "-emit-llvm-bc".* "-target-cpu" "(gfx908|gfx900)".* "-fopenmp".* "-x" "c".*}}

// SAN: {{"[^"]*llvm-offload-binary[^"]*" "-o".* "--image=file=.*.bc,triple=amdgcn-amd-amdhsa,arch=(gfx908|gfx1250|gfx1251)(:xnack\-|:xnack\+)?,kind=openmp(,feature=(\-xnack|\+xnack))?"}}
// SAN: {{"[^"]*clang[^"]*" "-cc1" "-triple" "x86_64-unknown-linux-gnu".* "-fopenmp".* "-fsanitize=address".* "--offload-targets=amdgcn-amd-amdhsa".* "-x" "ir".*}}
// SAN: {{"[^"]*clang-linker-wrapper[^"]*".* "--host-triple=x86_64-unknown-linux-gnu".* "--linker-path=[^"]*".* "--whole-archive" "[^"]*(libclang_rt.asan_static.a|libclang_rt.asan_static-x86_64.a)".* "--whole-archive" "[^"]*(libclang_rt.asan.a|libclang_rt.asan-x86_64.a)".*}}

// UNSUPPORTEDERROR: error: '-fsanitize=leak' option is not currently supported for target 'amdgcn-amd-amdhsa'
// XNACKERROR: error: '-fsanitize=address' option for offload arch 'gfx908:xnack-' is not currently supported there. Use it with an offload arch containing 'xnack+' instead
// INVALIDCOMBINATIONERROR: error: 'fuzzer' in '-fsanitize=fuzzer,address' option is not currently supported for target 'amdgcn-amd-amdhsa'

// WARNSANCOV: warning: ignoring '-fsanitize-coverage=inline-bool-flag' option as it is not currently supported for target 'amdgcn-amd-amdhsa'
// ERRSANCOV: error: '-fsanitize-coverage-stack-depth-callback-min=42' option is not currently supported for target 'amdgcn-amd-amdhsa'
