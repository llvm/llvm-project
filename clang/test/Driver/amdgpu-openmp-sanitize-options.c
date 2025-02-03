// REQUIRES: x86-registered-target, amdgpu-registered-target

// Fail on invalid ROCm Path.
// RUN:   not %clang -### -fopenmp --offload-arch=gfx908:xnack+ -fsanitize=address -fgpu-sanitize -nogpuinc --rocm-path=%S/Inputs/rocm-invalid  %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FAIL %s

// Enable multiple sanitizer's apart from ASan with invalid rocm-path.
// RUN:   not %clang -### -fopenmp --offload-arch=gfx908:xnack+ -fsanitize=address -fsanitize=leak -fgpu-sanitize --rocm-path=%S/Inputs/rocm-invalid -nogpuinc  %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=UNSUPPORTED,FAIL %s

// Memory, Leak, UndefinedBehaviour and Thread Sanitizer are not supported.
// RUN:   %clang -### -fopenmp --offload-arch=gfx908:xnack+ -fsanitize=address -fsanitize=leak -fgpu-sanitize --rocm-path=%S/Inputs/rocm -nogpuinc  %s 2>&1 \
// RUN:   | FileCheck --check-prefix=UNSUPPORTED %s


// ASan Enabled Test Cases
// ASan enabled for amdgpu-arch [gfx908]
// RUN:   %clang -### -fopenmp --offload-arch=gfx908 -fsanitize=address -fgpu-sanitize --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=NOXNACK,GPUSAN %s

// ASan enabled for amdgpu-arch [gfx908:xnack-]
// RUN:   %clang -### -fopenmp --offload-arch=gfx908:xnack- -fsanitize=address -fgpu-sanitize --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=XNACKNEG,GPUSAN %s

// ASan enabled for amdgpu-arch [gfx908:xnack+]
// RUN:   %clang -### -fopenmp --offload-arch=gfx908:xnack+ -fsanitize=address -fgpu-sanitize --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=GPUSAN %s

// ASan Disabled Test Cases
// ASan disabled for amdgpu-arch [gfx908]
// RUN:   %clang -### -fopenmp --offload-arch=gfx908 -fsanitize=address -fno-gpu-sanitize --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=NOGPUSAN %s

// ASan disabled for amdgpu-arch [gfx908:xnack-]
// RUN:   %clang -### -fopenmp --offload-arch=gfx908:xnack- -fsanitize=address -fno-gpu-sanitize --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=NOGPUSAN %s

// ASan disabled for amdgpu-arch [gfx908:xnack+]
// RUN:   %clang -### -fopenmp --offload-arch=gfx908:xnack+ -fsanitize=address -fno-gpu-sanitize --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=NOGPUSAN %s

// FAIL-DAG: error: cannot find ROCm device library for ABI version 5; provide its path via '--rocm-path' or '--rocm-device-lib-path', or pass '-nogpulib' to build without ROCm device library
// UNSUPPORTED-DAG: warning: ignoring '-fsanitize=leak' option as it is not currently supported for target 'amdgcn-amd-amdhsa'

// NOXNACK: warning: ignoring '-fsanitize=address' option for offload arch 'gfx908' as it is not currently supported there. Use it with an offload arch containing 'xnack+' instead
// XNACKNEG: warning: ignoring '-fsanitize=address' option for offload arch 'gfx908:xnack-' as it is not currently supported there. Use it with an offload arch containing 'xnack+' instead

// GPUSAN: {{"[^"]*clang[^"]*" "-cc1" "-triple" "x86_64-unknown-linux-gnu".* "-fopenmp".* "-fsanitize=address".* "-fopenmp-targets=amdgcn-amd-amdhsa".* "-x" "c".*}}
// GPUSAN: {{"[^"]*clang[^"]*" "-cc1" "-triple" "amdgcn-amd-amdhsa" "-aux-triple" "x86_64-unknown-linux-gnu".* "-emit-llvm-bc".* "-target-cpu" "gfx908".* "-fopenmp".* "-fsanitize=address".* "-x" "c".*}}
// GPUSAN: {{"[^"]*clang-offload-packager[^"]*" "-o".* "--image=file=.*.bc,triple=amdgcn-amd-amdhsa,arch=gfx908(:xnack\-|:xnack\+)?,kind=openmp(,feature=(\-xnack|\+xnack))?"}}
// GPUSAN: {{"[^"]*clang[^"]*" "-cc1" "-triple" "x86_64-unknown-linux-gnu".* "-fopenmp".* "-fsanitize=address".* "-fopenmp-targets=amdgcn-amd-amdhsa".* "-x" "ir".*}}
// GPUSAN: {{"[^"]*clang-linker-wrapper[^"]*" "--host-triple=x86_64-unknown-linux-gnu" "--linker-path=[^"]*ld.lld".* "--whole-archive" "[^"]*libclang_rt.asan_static.a".* "--whole-archive" "[^"]*libclang_rt.asan.a".*}}

// NOGPUSAN: {{"[^"]*clang[^"]*" "-cc1" "-triple" "x86_64-unknown-linux-gnu".* "-fopenmp".* "-fsanitize=address".* "-fopenmp-targets=amdgcn-amd-amdhsa".* "-x" "c".*}}
// NOGPUSAN: {{"[^"]*clang[^"]*" "-cc1" "-triple" "amdgcn-amd-amdhsa" "-aux-triple" "x86_64-unknown-linux-gnu".* "-emit-llvm-bc".* "-target-cpu" "gfx908".* "-fopenmp".* "-x" "c".*}}
// NOGPUSAN: {{"[^"]*clang-offload-packager[^"]*" "-o".* "--image=file=.*.bc,triple=amdgcn-amd-amdhsa,arch=gfx908(:xnack\-|:xnack\+)?,kind=openmp(,feature=(\-xnack|\+xnack))?"}}
// NOGPUSAN: {{"[^"]*clang[^"]*" "-cc1" "-triple" "x86_64-unknown-linux-gnu".* "-fopenmp".* "-fsanitize=address".* "-fopenmp-targets=amdgcn-amd-amdhsa".* "-x" "ir".*}}
// NOGPUSAN: {{"[^"]*clang-linker-wrapper[^"]*" "--host-triple=x86_64-unknown-linux-gnu" "--linker-path=[^"]*ld.lld".* "--whole-archive" "[^"]*libclang_rt.asan_static.a".* "--whole-archive" "[^"]*libclang_rt.asan.a".*}}
