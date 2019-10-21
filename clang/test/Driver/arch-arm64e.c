// Check the CPU defaults and overrides.

// RUN: %clang -arch arm64e -c %s -### 2>&1 | FileCheck %s --check-prefix VORTEX
// RUN: %clang -arch arm64e -mcpu=vortex -c %s -### 2>&1 | FileCheck %s --check-prefix VORTEX
// RUN: %clang -arch arm64e -mcpu=cyclone -c %s -### 2>&1 | FileCheck %s --check-prefix VORTEX
// RUN: %clang -arch arm64e -mcpu=lightning -c %s -### 2>&1 | FileCheck %s --check-prefix LIGHTNING
// VORTEX: "-cc1"{{.*}} "-target-cpu" "vortex"
// LIGHTNING: "-cc1"{{.*}} "-target-cpu" "lightning"
