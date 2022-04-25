// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN:   %clang -### -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908:xnack- -fsanitize=address -fgpu-sanitize --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=XNACK-DAG,GPUSAN,XNACKNEG %s

// RUN:   %clang -### -fopenmp --offload-arch=gfx908:xnack- -fsanitize=address -fgpu-sanitize --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=XNACK-DAG,GPUSAN,XNACKNEG %s

// RUN:   %clang -### -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908 -fsanitize=address -fgpu-sanitize --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=XNACKNONE-DAG,GPUSAN,XNACKNONE %s

// RUN:   %clang -### -fopenmp --offload-arch=gfx908 -fsanitize=address -fgpu-sanitize --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=XNACKNONE-DAG,GPUSAN,XNACKNONE %s

// RUN:   %clang -### -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908:xnack+ -fsanitize=address -fgpu-sanitize  --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=GPUSAN,XNACKPOS %s

// RUN:   %clang -### -fopenmp --offload-arch=gfx908:xnack+ -fsanitize=address -fgpu-sanitize --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=GPUSAN,XNACKPOS %s

// RUN:   %clang -### -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908:xnack+ -fsanitize=address -fno-gpu-sanitize --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=NOGPUSAN,XNACKPOS %s

// RUN:   %clang -### -fopenmp --offload-arch=gfx908:xnack+ -fsanitize=address -fno-gpu-sanitize --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=NOGPUSAN,XNACKPOS %s

// RUN:   %clang -### -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908:xnack- -fsanitize=address -fno-gpu-sanitize --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=NOGPUSAN,XNACKNEG %s

// RUN:   %clang -### -fopenmp --offload-arch=gfx908:xnack- -fsanitize=address -fno-gpu-sanitize  --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=NOGPUSAN,XNACKNEG %s

// XNACK-DAG: warning: ignoring '-fsanitize=address' option for offload arch 'gfx908:xnack-' as it is not currently supported there. Use it with an offload arch containing 'xnack+' instead
// XNACKNONE-DAG: warning: ignoring '-fsanitize=address' option for offload arch 'gfx908' as it is not currently supported there. Use it with an offload arch containing 'xnack+' instead

// GPUSAN: clang{{.*}}"-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-fopenmp"{{.*}}"-fsanitize=address"{{.*}}"-fopenmp-targets=amdgcn-amd-amdhsa"{{.*}}"-x" "c"{{.*}}
// GPUSAN: clang{{.*}}"-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-fopenmp"{{.*}}"-fsanitize=address"{{.*}}"-fopenmp-targets=amdgcn-amd-amdhsa"{{.*}}"-x" "ir"{{.*}}
// GPUSAN: clang{{.*}}"-cc1" "-triple" "amdgcn-amd-amdhsa" "-aux-triple" "x86_64-unknown-linux-gnu" "-emit-llvm-bc"{{.*}}"-target-cpu" "gfx908"{{.*}}"-fcuda-is-device"{{.*}}"-fopenmp"{{.*}}"-fsanitize=address"{{.*}}
// GPUSAN: llvm-link{{.*}}"--internalize" "--only-needed"{{.*}}"{{.*}}asanrtl.bc"{{.*}}"{{.*}}libomptarget-amdgcn-gfx908.bc"{{.*}}"-o" "{{.*}}.bc"

// NOGPUSAN: clang{{.*}}"-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-fopenmp"{{.*}}"-fsanitize=address"{{.*}}"-fopenmp-targets=amdgcn-amd-amdhsa"{{.*}}"-x" "c"{{.*}}
// NOGPUSAN: clang{{.*}}"-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-fopenmp"{{.*}}"-fsanitize=address"{{.*}}"-fopenmp-targets=amdgcn-amd-amdhsa"{{.*}}"-x" "ir"{{.*}}
// NOGPUSAN: clang{{.*}}"-cc1" "-triple" "amdgcn-amd-amdhsa" "-aux-triple" "x86_64-unknown-linux-gnu" "-emit-llvm-bc"{{.*}}"-target-cpu" "gfx908"{{.*}}"-fcuda-is-device"{{.*}}"-fopenmp"{{.*}}
// NOGPUSAN: llvm-link{{.*}}"--internalize" "--only-needed"{{.*}}"{{.*}}libomptarget-amdgcn-gfx908.bc"{{.*}}"-o" "{{.*}}.bc"

// XNACKNEG: opt{{.*}}"-mtriple=amdgcn-amd-amdhsa"{{.*}}"-mcpu=gfx908" "-mattr=-xnack"{{.*}}"-o" "{{.*}}.bc"
// XNACKNEG: llc{{.*}}"-mtriple=amdgcn-amd-amdhsa"{{.*}}"-mcpu=gfx908" "-filetype=obj"{{.*}}"-mattr=-xnack"{{.*}}"-o" "{{.*}}.o"
// XNACKNEG: lld{{.*}}"-flavor" "gnu" "--no-undefined" "-shared" "-o"{{.*}}.out{{.*}}"-plugin-opt=mcpu=gfx908" "-plugin-opt=-mattr=-xnack"
// XNACKNEG: clang-offload-wrapper{{.*}}"-target" "x86_64-unknown-linux-gnu"{{.*}}"--offload-arch=gfx908:xnack-"{{.*}}

// XNACKNONE: opt{{.*}}"-mtriple=amdgcn-amd-amdhsa"{{.*}}"-mcpu=gfx908"{{.*}}"-o" "{{.*}}.bc"
// XNACKNONE: llc{{.*}}"-mtriple=amdgcn-amd-amdhsa"{{.*}}"-mcpu=gfx908" "-filetype=obj"{{.*}}"-o" "{{.*}}.o"
// XNACKNONE: lld{{.*}}"-flavor" "gnu" "--no-undefined" "-shared" "-o"{{.*}}.out{{.*}}"-plugin-opt=mcpu=gfx908"
// XNACKNONE: clang-offload-wrapper{{.*}}"-target" "x86_64-unknown-linux-gnu"{{.*}}"--offload-arch=gfx908"{{.*}}

// XNACKPOS: opt{{.*}}"-mtriple=amdgcn-amd-amdhsa"{{.*}}"-mcpu=gfx908" "-mattr=+xnack"{{.*}}"-o" "{{.*}}.bc"
// XNACKPOS: llc{{.*}}"-mtriple=amdgcn-amd-amdhsa"{{.*}}"-mcpu=gfx908" "-filetype=obj"{{.*}}"-mattr=+xnack"{{.*}}"-o" "{{.*}}.o"
// XNACKPOS: lld{{.*}}"-flavor" "gnu" "--no-undefined" "-shared" "-o"{{.*}}.out{{.*}}"-plugin-opt=mcpu=gfx908" "-plugin-opt=-mattr=+xnack"
// XNACKPOS: clang-offload-wrapper{{.*}}"-target" "x86_64-unknown-linux-gnu"{{.*}}"--offload-arch=gfx908:xnack+"{{.*}}

// RUN:   %clang -### -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908:xnack+ -fsanitize=address -fgpu-sanitize -nogpuinc --rocm-path=%S/Inputs/rocm-invalid  %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FAIL %s

// FAIL: error: AMDGPU address sanitizer runtime library (asanrtl) not found. Please install ROCm device library which supports address sanitizer
