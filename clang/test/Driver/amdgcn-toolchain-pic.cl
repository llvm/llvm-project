// RUN: %clang -### --target=amdgcn-- -mcpu=gfx803 %s 2>&1 | FileCheck %s
// RUN: %clang -### --target=amdgcn-amd- -mcpu=gfx803 %s 2>&1 | FileCheck %s
// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx803 %s 2>&1 | FileCheck %s
// RUN: %clang -### --target=amdgcn-amd-amdpal -mcpu=gfx803 %s 2>&1 | FileCheck %s
// RUN: %clang -### --target=amdgcn-amd-mesa3d -mcpu=gfx803 %s 2>&1 | FileCheck %s

// CHECK: clang{{.*}} "-mrelocation-model" "pic" "-pic-level" "1"
