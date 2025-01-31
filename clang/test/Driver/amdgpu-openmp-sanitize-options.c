// REQUIRES: x86-registered-target, amdgpu-registered-target

// Fail on invalid ROCm Path.
// RUN:   not %clang -### --target=x86_64-unknown-linux-gnu -fopenmp --offload-arch=gfx908:xnack+ -fsanitize=address -fgpu-sanitize -nogpuinc --rocm-path=%S/Inputs/rocm-invalid  %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FAIL %s

// Enable multiple sanitizer's apart from ASan with invalid rocm-path.
// RUN:   not %clang -### --target=x86_64-unknown-linux-gnu -fopenmp --offload-arch=gfx908:xnack+ -fsanitize=address -fsanitize=leak -fgpu-sanitize --rocm-path=%S/Inputs/rocm-invalid -nogpuinc  %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NOTSUPPORTED,FAIL %s

// Memory, Leak, UndefinedBehaviour and Thread Sanitizer are not supported on AMDGPU.
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp --offload-arch=gfx908:xnack+ -fsanitize=address -fsanitize=leak -fgpu-sanitize --rocm-path=%S/Inputs/rocm -nogpuinc  %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NOTSUPPORTED %s

// GPU ASan Enabled Test Cases
// ASan enabled for amdgpu-arch [gfx908]
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp --offload-arch=gfx908 -fsanitize=address -fgpu-sanitize --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=NOXNACK,GPUSAN %s

// GPU ASan enabled for amdgpu-arch [gfx908:xnack-]
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp --offload-arch=gfx908:xnack- -fsanitize=address -fgpu-sanitize --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=XNACKNEG,GPUSAN %s

// GPU ASan enabled for amdgpu-arch [gfx908:xnack+]
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp --offload-arch=gfx908:xnack+ -fsanitize=address -fgpu-sanitize --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=GPUSAN %s

// ASan enabled for multiple amdgpu-arch [gfx908:xnack+,gfx900:xnack+]
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp --offload-arch=gfx908:xnack+ --offload-arch=gfx900:xnack+ -fsanitize=address -fgpu-sanitize --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=GPUSAN %s

// GPU ASan Disabled Test Cases
// ASan disabled for amdgpu-arch [gfx908]
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp --offload-arch=gfx908 -fsanitize=address -fno-gpu-sanitize --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=NOGPUSAN %s

// GPU ASan disabled for amdgpu-arch [gfx908:xnack-]
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp --offload-arch=gfx908:xnack- -fsanitize=address -fno-gpu-sanitize --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=NOGPUSAN %s

// GPU ASan disabled for amdgpu-arch [gfx908:xnack+]
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp --offload-arch=gfx908:xnack+ -fsanitize=address -fno-gpu-sanitize --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=NOGPUSAN %s

// ASan disabled for amdgpu-arch [gfx908:xnack+,gfx900:xnack+]
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp --offload-arch=gfx908:xnack+ --offload-arch=gfx900:xnack+ -fsanitize=address -fno-gpu-sanitize --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=NOGPUSAN %s

// FAIL-DAG: error: cannot find ROCm device library for ABI version 5; provide its path via '--rocm-path' or '--rocm-device-lib-path', or pass '-nogpulib' to build without ROCm device library
// NOTSUPPORTED-DAG: warning: ignoring '-fsanitize=leak' option as it is not currently supported for target 'amdgcn-amd-amdhsa'

// NOXNACK: warning: ignoring '-fsanitize=address' option for offload arch 'gfx908' as it is not currently supported there. Use it with an offload arch containing 'xnack+' instead
// XNACKNEG: warning: ignoring '-fsanitize=address' option for offload arch 'gfx908:xnack-' as it is not currently supported there. Use it with an offload arch containing 'xnack+' instead

// GPUSAN: {{"[^"]*clang[^"]*" "-cc1" "-triple" "x86_64-unknown-linux-gnu".* "-fopenmp".* "-fsanitize=address".* "-fopenmp-targets=amdgcn-amd-amdhsa".* "-x" "c".*}}
// GPUSAN: {{"[^"]*clang[^"]*" "-cc1" "-triple" "amdgcn-amd-amdhsa" "-aux-triple" "x86_64-unknown-linux-gnu".* "-emit-llvm-bc".* "-target-cpu" "(gfx908|gfx900)".* "-fopenmp".* "-fsanitize=address".* "-x" "c".*}}
// GPUSAN: {{"[^"]*clang-offload-packager[^"]*" "-o".* "--image=file=.*.bc,triple=amdgcn-amd-amdhsa,arch=gfx908(:xnack\-|:xnack\+)?,kind=openmp(,feature=(\-xnack|\+xnack))?"}}
// GPUSAN: {{"[^"]*clang[^"]*" "-cc1" "-triple" "x86_64-unknown-linux-gnu".* "-fopenmp".* "-fsanitize=address".* "-fopenmp-targets=amdgcn-amd-amdhsa".* "-x" "ir".*}}
// GPUSAN: {{"[^"]*clang-linker-wrapper[^"]*" "--host-triple=x86_64-unknown-linux-gnu" "--linker-path=[^"]*".* "--whole-archive" "[^"]*libclang_rt.asan_static.a".* "--whole-archive" "[^"]*libclang_rt.asan.a".*}}

// NOGPUSAN: {{"[^"]*clang[^"]*" "-cc1" "-triple" "x86_64-unknown-linux-gnu".* "-fopenmp".* "-fsanitize=address".* "-fopenmp-targets=amdgcn-amd-amdhsa".* "-x" "c".*}}
// NOGPUSAN: {{"[^"]*clang[^"]*" "-cc1" "-triple" "amdgcn-amd-amdhsa" "-aux-triple" "x86_64-unknown-linux-gnu".* "-emit-llvm-bc".* "-target-cpu" "(gfx908|gfx900)".* "-fopenmp".* "-x" "c".*}}
// NOGPUSAN: {{"[^"]*clang-offload-packager[^"]*" "-o".* "--image=file=.*.bc,triple=amdgcn-amd-amdhsa,arch=gfx908(:xnack\-|:xnack\+)?,kind=openmp(,feature=(\-xnack|\+xnack))?"}}
// NOGPUSAN: {{"[^"]*clang[^"]*" "-cc1" "-triple" "x86_64-unknown-linux-gnu".* "-fopenmp".* "-fsanitize=address".* "-fopenmp-targets=amdgcn-amd-amdhsa".* "-x" "ir".*}}
// NOGPUSAN: {{"[^"]*clang-linker-wrapper[^"]*" "--host-triple=x86_64-unknown-linux-gnu" "--linker-path=[^"]*".* "--whole-archive" "[^"]*libclang_rt.asan_static.a".* "--whole-archive" "[^"]*libclang_rt.asan.a".*}}
