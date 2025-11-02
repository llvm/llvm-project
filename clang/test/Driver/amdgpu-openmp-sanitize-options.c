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

// ASan enabled for multiple amdgpu-arch [gfx908:xnack+,gfx900:xnack+]
// RUN:   %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=gfx908:xnack+ --offload-arch=gfx900:xnack+ -fsanitize=address -fgpu-sanitize --rocm-path=%S/Inputs/rocm %s 2>&1 \
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

// FAIL-DAG: error: cannot find ROCm device library for ABI version 5; provide its path via '--rocm-path' or '--rocm-device-lib-path', or pass '-nogpulib' to build without ROCm device library
// NOTSUPPORTED-DAG: warning: ignoring '-fsanitize=leak' option as it is not currently supported for target 'amdgcn-amd-amdhsa'

// NOXNACK: warning: ignoring '-fsanitize=address' option for offload arch 'gfx908' as it is not currently supported there. Use it with an offload arch containing 'xnack+' instead
// XNACKNEG: warning: ignoring '-fsanitize=address' option for offload arch 'gfx908:xnack-' as it is not currently supported there. Use it with an offload arch containing 'xnack+' instead

// HOSTSAN: {{"[^"]*clang[^"]*" "-cc1" "-triple" "x86_64-unknown-linux-gnu".* "-fopenmp".* "-fsanitize=address".* "--offload-targets=amdgcn-amd-amdhsa".* "-x" "c".*}}

// GPUSAN: {{"[^"]*clang[^"]*" "-cc1" "-triple" "amdgcn-amd-amdhsa" "-aux-triple" "x86_64-unknown-linux-gnu".* "-emit-llvm-bc".* "-mlink-bitcode-file" "[^"]*asanrtl.bc".* "-mlink-bitcode-file" "[^"]*ockl.bc".* "-target-cpu" "(gfx908|gfx900)".* "-fopenmp".* "-fsanitize=address".* "-x" "c".*}}
// NOGPUSAN: {{"[^"]*clang[^"]*" "-cc1" "-triple" "amdgcn-amd-amdhsa" "-aux-triple" "x86_64-unknown-linux-gnu".* "-emit-llvm-bc".* "-target-cpu" "(gfx908|gfx900)".* "-fopenmp".* "-x" "c".*}}

// SAN: {{"[^"]*llvm-offload-binary[^"]*" "-o".* "--image=file=.*.bc,triple=amdgcn-amd-amdhsa,arch=gfx908(:xnack\-|:xnack\+)?,kind=openmp(,feature=(\-xnack|\+xnack))?"}}
// SAN: {{"[^"]*clang[^"]*" "-cc1" "-triple" "x86_64-unknown-linux-gnu".* "-fopenmp".* "-fsanitize=address".* "--offload-targets=amdgcn-amd-amdhsa".* "-x" "ir".*}}
// SAN: {{"[^"]*clang-linker-wrapper[^"]*".* "--host-triple=x86_64-unknown-linux-gnu".* "--linker-path=[^"]*".* "--whole-archive" "[^"]*(libclang_rt.asan_static.a|libclang_rt.asan_static-x86_64.a)".* "--whole-archive" "[^"]*(libclang_rt.asan.a|libclang_rt.asan-x86_64.a)".*}}
