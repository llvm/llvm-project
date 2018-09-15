// REQUIRES: clang-driver
//
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 %s -g -O0 --no-cuda-noopt-device-debug 2>&1 | \
// RUN:   FileCheck %s -check-prefix NO_DEBUG
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 %s -g -O3 2>&1 | \
// RUN:   FileCheck %s -check-prefix NO_DEBUG
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 %s -g -O3 --no-cuda-noopt-device-debug 2>&1 | \
// RUN:   FileCheck %s -check-prefix NO_DEBUG
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 %s -g0 2>&1 | \
// RUN:   FileCheck %s -check-prefix NO_DEBUG
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 %s -ggdb0 -O3 --cuda-noopt-device-debug 2>&1 | \
// RUN:   FileCheck %s -check-prefix NO_DEBUG
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 %s -ggdb1 2>&1 | \
// RUN:   FileCheck %s -check-prefix NO_DEBUG -check-prefix LINE_TABLE
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 %s -gline-tables-only -O2 --cuda-noopt-device-debug 2>&1 | \
// RUN:   FileCheck %s -check-prefix NO_DEBUG -check-prefix LINE_TABLE

// NO_DEBUG-NOT: warning: debug
// LINE_TABLE-NOT: warning: debug
// NO_DEBUG: ptxas
// NO_DEBUG-NOT: "-g"
// LINE_TABLE: "-lineinfo"
// NO_DEBUG: fatbinary
// NO_DEBUG-NOT: "-g"

// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 %s -g 2>&1 | \
// RUN:   FileCheck %s -check-prefix HAS_DEBUG
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 %s -g -O0 --cuda-noopt-device-debug 2>&1 | \
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

// HAS_DEBUG-NOT: warning: debug
// HAS_DEBUG: "-fcuda-is-device"
// HAS_DEBUG-SAME: "-dwarf-version=2"
// HAS_DEBUG: ptxas
// HAS_DEBUG-SAME: "-g"
// HAS_DEBUG-SAME: "--dont-merge-basicblocks"
// HAS_DEBUG-SAME: "--return-at-end"
// HAS_DEBUG: fatbinary
// HAS_DEBUG-SAME: "-g"

