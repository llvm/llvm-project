// Tests that "sm_XX" gets correctly converted to "compute_YY" when we invoke
// fatbinary.
//
// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=COMMON,SM20 %s
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_21 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=COMMON,SM21 %s
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_30 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=COMMON,SM30 %s
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_32 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=COMMON,SM32 %s
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_35 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=COMMON,SM35 %s
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_37 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=COMMON,SM37 %s
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_50 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=COMMON,SM50 %s
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_52 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=COMMON,SM52 %s
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_53 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=COMMON,SM53 %s
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_60 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=COMMON,SM60 %s
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_61 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=COMMON,SM61 %s
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_62 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=COMMON,SM62 %s
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_70 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=COMMON,SM70 %s

// COMMON: ptxas
// COMMON-SAME: -m64
// COMMON: fatbinary

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
