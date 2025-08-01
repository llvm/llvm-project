// Tests that "sm_XX" gets correctly converted to "compute_YY" when we invoke
// fatbinary.

// RUN: %clang -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM20 %s
// RUN: %clang -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=sm_21 --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM21 %s
// RUN: %clang -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=sm_30 --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM30 %s
// RUN: %clang -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=sm_32 --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM32 %s
// RUN: %clang -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=sm_35 --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM35 %s
// RUN: %clang -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=sm_37 --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM37 %s
// RUN: %clang -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=sm_50 --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM50 %s
// RUN: %clang -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=sm_52 --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM52 %s
// RUN: %clang -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=sm_53 --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM53 %s
// RUN: %clang -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=sm_60 --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM60 %s
// RUN: %clang -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=sm_61 --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM61 %s
// RUN: %clang -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=sm_62 --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM62 %s
// RUN: %clang -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=sm_70 --cuda-path=%S/Inputs/CUDA_111/usr/local/cuda %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM70 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx600 -nogpuinc -nogpulib %s 2>&1 \
// RUN: | FileCheck -check-prefixes=HIP,GFX600 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx601 -nogpuinc -nogpulib %s 2>&1 \
// RUN: | FileCheck -check-prefixes=HIP,GFX601 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx602 -nogpuinc -nogpulib %s 2>&1 \
// RUN: | FileCheck -check-prefixes=HIP,GFX602 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx700 -nogpuinc -nogpulib %s 2>&1 \
// RUN: | FileCheck -check-prefixes=HIP,GFX700 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx701 -nogpuinc -nogpulib %s 2>&1 \
// RUN: | FileCheck -check-prefixes=HIP,GFX701 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx702 -nogpuinc -nogpulib %s 2>&1 \
// RUN: | FileCheck -check-prefixes=HIP,GFX702 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx703 -nogpuinc -nogpulib %s 2>&1 \
// RUN: | FileCheck -check-prefixes=HIP,GFX703 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx704 -nogpuinc -nogpulib %s 2>&1 \
// RUN: | FileCheck -check-prefixes=HIP,GFX704 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx705 -nogpuinc -nogpulib %s 2>&1 \
// RUN: | FileCheck -check-prefixes=HIP,GFX705 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx801 -nogpuinc -nogpulib %s 2>&1 \
// RUN: | FileCheck -check-prefixes=HIP,GFX801 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx802 -nogpuinc -nogpulib %s 2>&1 \
// RUN: | FileCheck -check-prefixes=HIP,GFX802 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx803 -nogpuinc -nogpulib %s 2>&1 \
// RUN: | FileCheck -check-prefixes=HIP,GFX803 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx805 -nogpuinc -nogpulib %s 2>&1 \
// RUN: | FileCheck -check-prefixes=HIP,GFX805 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx810 -nogpuinc -nogpulib %s 2>&1 \
// RUN: | FileCheck -check-prefixes=HIP,GFX810 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx900 -nogpuinc -nogpulib %s 2>&1 \
// RUN: | FileCheck -check-prefixes=HIP,GFX900 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx902 -nogpuinc -nogpulib %s 2>&1 \
// RUN: | FileCheck -check-prefixes=HIP,GFX902 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=amdgcnspirv -nogpuinc -nogpulib %s 2>&1 \
// RUN: | FileCheck -check-prefixes=HIP,SPIRV %s

// CUDA: ptxas
// CUDA-SAME: -m64
// CUDA: fatbinary

// HIP: clang-offload-packager

// SM20:--image=profile=sm_20{{.*}}
// SM21:--image=profile=sm_21{{.*}}
// SM30:--image=profile=sm_30{{.*}}
// SM32:--image=profile=sm_32{{.*}}
// SM35:--image=profile=sm_35{{.*}}
// SM37:--image=profile=sm_37{{.*}}
// SM50:--image=profile=sm_50{{.*}}
// SM52:--image=profile=sm_52{{.*}}
// SM53:--image=profile=sm_53{{.*}}
// SM60:--image=profile=sm_60{{.*}}
// SM61:--image=profile=sm_61{{.*}}
// SM62:--image=profile=sm_62{{.*}}
// SM70:--image=profile=sm_70{{.*}}
// GFX600:triple=amdgcn-amd-amdhsa,arch=gfx600
// GFX601:triple=amdgcn-amd-amdhsa,arch=gfx601
// GFX602:triple=amdgcn-amd-amdhsa,arch=gfx602
// GFX700:triple=amdgcn-amd-amdhsa,arch=gfx700
// GFX701:triple=amdgcn-amd-amdhsa,arch=gfx701
// GFX702:triple=amdgcn-amd-amdhsa,arch=gfx702
// GFX703:triple=amdgcn-amd-amdhsa,arch=gfx703
// GFX704:triple=amdgcn-amd-amdhsa,arch=gfx704
// GFX705:triple=amdgcn-amd-amdhsa,arch=gfx705
// GFX801:triple=amdgcn-amd-amdhsa,arch=gfx801
// GFX802:triple=amdgcn-amd-amdhsa,arch=gfx802
// GFX803:triple=amdgcn-amd-amdhsa,arch=gfx803
// GFX805:triple=amdgcn-amd-amdhsa,arch=gfx805
// GFX810:triple=amdgcn-amd-amdhsa,arch=gfx810
// GFX900:triple=amdgcn-amd-amdhsa,arch=gfx900
// GFX902:triple=amdgcn-amd-amdhsa,arch=gfx902
// SPIRV:triple=spirv64-amd-amdhsa,arch=amdgcnspirv
