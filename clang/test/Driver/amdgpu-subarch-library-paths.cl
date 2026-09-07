// Test that the AMDGPU toolchain emits a resource-directory -L for the
// arch-specific library path when using a subarch triple. The current triple's
// directory is searched first, then the subarch-stripped "amdgpu" and legacy
// "amdgcn" directories.

// The directory named for the current triple is searched.
// RUN:   %clang -### --target=amdgpu10.30-amd-amdhsa -nogpulib \
// RUN:     -resource-dir=%S/Inputs/resource_dir_amdgpu_triples %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CURRENT-TRIPLE %s

// A subarch triple falls back to the subarch-stripped "amdgpu" directory.
// RUN:   %clang -### --target=amdgpu9.0a-amd-amdhsa -nogpulib \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck -check-prefix=FALLBACK-AMDGPU %s

// A subarch triple falls back to the legacy "amdgcn" directory.
// RUN:   %clang -### --target=amdgpu9.0a-amd-amdhsa -nogpulib \
// RUN:     -resource-dir=%S/Inputs/resource_dir_amdgpu_legacy_triple %s 2>&1 \
// RUN:   | FileCheck -check-prefix=FALLBACK-AMDGCN %s

// CURRENT-TRIPLE: "-L{{.*}}resource_dir_amdgpu_triples{{[/\\]+}}lib{{[/\\]+}}amdgpu10.30-amd-amdhsa"

// FALLBACK-AMDGPU: "-L{{.*}}resource_dir_with_amdgpu_per_target_subdir{{[/\\]+}}lib{{[/\\]+}}amdgpu-amd-amdhsa"

// FALLBACK-AMDGCN: "-L{{.*}}resource_dir_amdgpu_legacy_triple{{[/\\]+}}lib{{[/\\]+}}amdgcn-amd-amdhsa"
