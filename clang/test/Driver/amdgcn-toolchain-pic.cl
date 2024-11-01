// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx803 %s 2>&1 | FileCheck %s
// RUN: %clang -### --target=amdgcn-amd-amdpal -mcpu=gfx803 %s 2>&1 | FileCheck %s
// RUN: %clang -### --target=amdgcn-amd-mesa3d -mcpu=gfx803 %s 2>&1 | FileCheck %s

// CHECK: "-cc1"{{.*}} "-mrelocation-model" "pic" "-pic-level" "2"
