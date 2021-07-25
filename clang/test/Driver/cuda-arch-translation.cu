// Tests that "sm_XX" gets correctly converted to "compute_YY" when we invoke
// fatbinary.
//
// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM20 %s
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_21 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM21 %s
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_30 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM30 %s
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_32 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM32 %s
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_35 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM35 %s
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_37 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM37 %s
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_50 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM50 %s
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_52 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM52 %s
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_53 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM53 %s
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_60 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM60 %s
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_61 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM61 %s
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_62 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM62 %s
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_70 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM70 %s
// RUN: %clang -x hip -### -target x86_64-linux-gnu -c --cuda-gpu-arch=gfx600 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=HIP,GFX600 %s
// RUN: %clang -x hip -### -target x86_64-linux-gnu -c --cuda-gpu-arch=gfx601 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=HIP,GFX601 %s
// RUN: %clang -x hip -### -target x86_64-linux-gnu -c --cuda-gpu-arch=gfx602 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=HIP,GFX602 %s
// RUN: %clang -x hip -### -target x86_64-linux-gnu -c --cuda-gpu-arch=gfx700 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=HIP,GFX700 %s
// RUN: %clang -x hip -### -target x86_64-linux-gnu -c --cuda-gpu-arch=gfx701 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=HIP,GFX701 %s
// RUN: %clang -x hip -### -target x86_64-linux-gnu -c --cuda-gpu-arch=gfx702 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=HIP,GFX702 %s
// RUN: %clang -x hip -### -target x86_64-linux-gnu -c --cuda-gpu-arch=gfx703 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=HIP,GFX703 %s
// RUN: %clang -x hip -### -target x86_64-linux-gnu -c --cuda-gpu-arch=gfx704 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=HIP,GFX704 %s
// RUN: %clang -x hip -### -target x86_64-linux-gnu -c --cuda-gpu-arch=gfx705 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=HIP,GFX705 %s
// RUN: %clang -x hip -### -target x86_64-linux-gnu -c --cuda-gpu-arch=gfx801 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=HIP,GFX801 %s
// RUN: %clang -x hip -### -target x86_64-linux-gnu -c --cuda-gpu-arch=gfx802 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=HIP,GFX802 %s
// RUN: %clang -x hip -### -target x86_64-linux-gnu -c --cuda-gpu-arch=gfx803 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=HIP,GFX803 %s
// RUN: %clang -x hip -### -target x86_64-linux-gnu -c --cuda-gpu-arch=gfx805 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=HIP,GFX805 %s
// RUN: %clang -x hip -### -target x86_64-linux-gnu -c --cuda-gpu-arch=gfx810 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=HIP,GFX810 %s
// RUN: %clang -x hip -### -target x86_64-linux-gnu -c --cuda-gpu-arch=gfx900 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=HIP,GFX900 %s
// RUN: %clang -x hip -### -target x86_64-linux-gnu -c --cuda-gpu-arch=gfx902 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=HIP,GFX902 %s

// CUDA: ptxas
// CUDA-SAME: -m64
// CUDA: fatbinary

// HIP: clang-offload-bundler

// SM20:--image=profile=sm_20{{.*}}--image=profile=compute_20
// SM21:--image=profile=sm_21{{.*}}--image=profile=compute_20
// SM30:--image=profile=sm_30{{.*}}--image=profile=compute_30
// SM32:--image=profile=sm_32{{.*}}--image=profile=compute_32
// SM35:--image=profile=sm_35{{.*}}--image=profile=compute_35
// SM37:--image=profile=sm_37{{.*}}--image=profile=compute_37
// SM50:--image=profile=sm_50{{.*}}--image=profile=compute_50
// SM52:--image=profile=sm_52{{.*}}--image=profile=compute_52
// SM53:--image=profile=sm_53{{.*}}--image=profile=compute_53
// SM60:--image=profile=sm_60{{.*}}--image=profile=compute_60
// SM61:--image=profile=sm_61{{.*}}--image=profile=compute_61
// SM62:--image=profile=sm_62{{.*}}--image=profile=compute_62
// SM70:--image=profile=sm_70{{.*}}--image=profile=compute_70
// GFX600:-targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx600
// GFX601:-targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx601
// GFX602:-targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx602
// GFX700:-targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx700
// GFX701:-targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx701
// GFX702:-targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx702
// GFX703:-targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx703
// GFX704:-targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx704
// GFX705:-targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx705
// GFX801:-targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx801
// GFX802:-targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx802
// GFX803:-targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx803
// GFX805:-targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx805
// GFX810:-targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx810
// GFX900:-targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx900
// GFX902:-targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx902
