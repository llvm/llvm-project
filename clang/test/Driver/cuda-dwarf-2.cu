
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 %s -g -O1 --no-cuda-noopt-device-debug 2>&1 | \
// RUN:   FileCheck %s -check-prefix DEBUG_DIRECTIVES
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 %s -g -O3 2>&1 | \
// RUN:   FileCheck %s -check-prefix DEBUG_DIRECTIVES
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 %s -g -O3 --no-cuda-noopt-device-debug 2>&1 | \
// RUN:   FileCheck %s -check-prefix DEBUG_DIRECTIVES
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 %s -g0 2>&1 | \
// RUN:   FileCheck %s -check-prefix NO_DEBUG
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 %s -ggdb0 -O3 --cuda-noopt-device-debug 2>&1 | \
// RUN:   FileCheck %s -check-prefix NO_DEBUG
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 %s -gline-directives-only -O2 --cuda-noopt-device-debug 2>&1 | \
// RUN:   FileCheck %s -check-prefix DEBUG_DIRECTIVES

// NO_DEBUG-NOT: warning: debug
// DEBUG_DIRECTIVES-NOT: warning: debug
// NO_DEBUG: "-fcuda-is-device"
// NO_DEBUG-NOT: "-debug-info-kind=
// NO_DEBUG: ptxas
// NO_DEBUG-NOT: "-g"
// DEBUG_DIRECTIVES: "-fcuda-is-device"
// DEBUG_DIRECTIVES-SAME: "-debug-info-kind=line-directives-only"
// DEBUG_DIRECTIVES: ptxas
// DEBUG_DIRECTIVES-SAME: "-lineinfo"
// NO_DEBUG: fatbinary
// NO_DEBUG-NOT: "-g"

// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 %s -g 2>&1 | \
// RUN:   FileCheck %s -check-prefix HAS_DEBUG
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 %s -g -O0 --cuda-noopt-device-debug 2>&1 | \
// RUN:   FileCheck %s -check-prefix HAS_DEBUG
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 %s -g -O0 --no-cuda-noopt-device-debug 2>&1 | \
// RUN:   FileCheck %s -check-prefix HAS_DEBUG
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 %s -g -O3 --cuda-noopt-device-debug 2>&1 | \
// RUN:   FileCheck %s -check-prefix HAS_DEBUG
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 %s -g2 2>&1 | \
// RUN:   FileCheck %s -check-prefix HAS_DEBUG
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 %s -ggdb2 -O0 2>&1 | \
// RUN:   FileCheck %s -check-prefix HAS_DEBUG
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 %s -g3 -O2 --cuda-noopt-device-debug 2>&1 | \
// RUN:   FileCheck %s -check-prefix HAS_DEBUG
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 %s -ggdb3 -O3 --cuda-noopt-device-debug 2>&1 | \
// RUN:   FileCheck %s -check-prefix HAS_DEBUG
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 %s -ggdb1 2>&1 | \
// RUN:   FileCheck %s -check-prefix HAS_DEBUG
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 %s -gline-tables-only -O2 --cuda-noopt-device-debug 2>&1 | \
// RUN:   FileCheck %s -check-prefix HAS_DEBUG

// HAS_DEBUG-NOT: warning: debug
// HAS_DEBUG: "-fcuda-is-device"
// HAS_DEBUG-SAME: "-debug-info-kind={{constructor|line-tables-only}}"
// HAS_DEBUG-SAME: "-dwarf-version=2"
// HAS_DEBUG: ptxas
// HAS_DEBUG-SAME: "-g"
// HAS_DEBUG-SAME: "--dont-merge-basicblocks"
// HAS_DEBUG-SAME: "--return-at-end"
// HAS_DEBUG: fatbinary
// HAS_DEBUG-SAME: "-g"

