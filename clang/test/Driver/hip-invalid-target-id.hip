// RUN: not %clang -### --target=x86_64-linux-gnu \
// RUN:   -x hip --offload-arch=gfx908 \
// RUN:   --offload-arch=gfx908xnack \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefix=NOPLUS %s

// NOPLUS: error: unsupported HIP gpu architecture: gfx908xnack 

// RUN: not %clang -### --target=x86_64-linux-gnu \
// RUN:   -x hip --offload-arch=gfx900 \
// RUN:   --offload-arch=gfx908:xnack+:xnack+ \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefix=ORDER %s

// ORDER: error: invalid target ID 'gfx908:xnack+:xnack+'

// RUN: not %clang -### --target=x86_64-linux-gnu \
// RUN:   -x hip --offload-arch=gfx908 \
// RUN:   --offload-arch=gfx908:unknown+ \
// RUN:   --offload-arch=gfx908+sramecc+unknown \
// RUN:   --offload-arch=gfx900+xnack \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefix=UNK %s

// UNK: error: unsupported HIP gpu architecture: gfx900+xnack 

// RUN: not %clang -### --target=x86_64-linux-gnu \
// RUN:   -x hip --offload-arch=gfx908 \
// RUN:   --offload-arch=gfx908:sramecc+:unknown+ \
// RUN:   --offload-arch=gfx900+xnack \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefix=MIXED %s

// MIXED: error: unsupported HIP gpu architecture: gfx900+xnack 

// RUN: not %clang -### --target=x86_64-linux-gnu \
// RUN:   -x hip --offload-arch=gfx908 \
// RUN:   --offload-arch=gfx900:sramecc+ \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefix=UNSUP %s

// UNSUP: error: invalid target ID 'gfx900:sramecc+'

/ RUN: not %clang -### --target=x86_64-linux-gnu \
// RUN:   -x hip --offload-arch=gfx908 \
// RUN:   --offload-arch=gfx900:xnack \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefix=NOSIGN %s

// NOSIGN: error: invalid target ID 'gfx900:xnack'

// RUN: not %clang -### --target=x86_64-linux-gnu \
// RUN:   -x hip --offload-arch=gfx908 \
// RUN:   --offload-arch=gfx900+xnack \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefix=NOCOLON %s

// NOCOLON: error: unsupported HIP gpu architecture: gfx900+xnack

// RUN: not %clang -### --target=x86_64-linux-gnu \
// RUN:   -x hip --offload-arch=gfx908 \
// RUN:   --offload-arch=gfx908:xnack+ \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefix=COMBO %s

// COMBO: error: invalid offload arch combinations: 'gfx908' and 'gfx908:xnack+'
