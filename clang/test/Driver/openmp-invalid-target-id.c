// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

//
// Legacy mode (-fopenmp-targets,-Xopenmp-target,-march) tests for TargetID
//
// RUN: not %clang -### -target x86_64-linux-gnu -fopenmp\
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa,amdgcn-amd-amdhsa \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908 \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908xnack \
// RUN:   %s 2>&1 | FileCheck -check-prefix=NOPLUS-L %s

// NOPLUS-L: error: invalid target ID 'gfx908xnack'

// RUN: not %clang -### -target x86_64-linux-gnu -fopenmp\
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa,amdgcn-amd-amdhsa \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx900 \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908:xnack+:xnack+ \
// RUN:   %s 2>&1 | FileCheck -check-prefix=ORDER-L %s

// ORDER-L: error: invalid target ID 'gfx908:xnack+:xnack+'

// RUN: not %clang -### -target x86_64-linux-gnu -fopenmp\
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa,amdgcn-amd-amdhsa,amdgcn-amd-amdhsa,amdgcn-amd-amdhsa \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908 \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908:unknown+ \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908+sramecc+unknown \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx900+xnack \
// RUN:   %s 2>&1 | FileCheck -check-prefix=UNK-L %s

// UNK-L: error: invalid target ID 'gfx908:unknown+'

// RUN: not %clang -### -target x86_64-linux-gnu -fopenmp\
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa,amdgcn-amd-amdhsa,amdgcn-amd-amdhsa \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908 \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908:sramecc+:unknown+ \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx900+xnack \
// RUN:   %s 2>&1 | FileCheck -check-prefix=MIXED-L %s

// MIXED-L: error: invalid target ID 'gfx908:sramecc+:unknown+'

// RUN: not %clang -### -target x86_64-linux-gnu -fopenmp\
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa,amdgcn-amd-amdhsa \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908 \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx900:sramecc+ \
// RUN:   %s 2>&1 | FileCheck -check-prefix=UNSUP-L %s

// UNSUP-L: error: invalid target ID 'gfx900:sramecc+'

// RUN: not %clang -### -target x86_64-linux-gnu -fopenmp\
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa,amdgcn-amd-amdhsa \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908 \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx900:xnack \
// RUN:   %s 2>&1 | FileCheck -check-prefix=NOSIGN-L %s

// NOSIGN-L: error: invalid target ID 'gfx900:xnack'

// RUN: not %clang -### -target x86_64-linux-gnu -fopenmp\
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa,amdgcn-amd-amdhsa \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908 \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx900+xnack \
// RUN:   %s 2>&1 | FileCheck -check-prefix=NOCOLON-L %s

// NOCOLON-L: error: invalid target ID 'gfx900+xnack'

// RUN: not %clang -### -target x86_64-linux-gnu -fopenmp\
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa,amdgcn-amd-amdhsa \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908 \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908:xnack+ \
// RUN:   %s 2>&1 | FileCheck -check-prefix=COMBO-L %s

// COMBO-L: error: invalid offload arch combinations: 'gfx908' and 'gfx908:xnack+'
