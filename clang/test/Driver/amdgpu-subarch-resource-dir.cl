// RUN:   %clang -### --target=amdgpu10.30-amd-amdhsa-llvm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_amdgpu_subarch_triples %s 2>&1 \
// RUN:   | FileCheck -check-prefix=TOPLEVEL-SUBARCH %s


// RUN:   %clang -### --target=amdgpu9.0a-amd-amdhsa-llvm -mcpu=gfx90a \
// RUN:     -resource-dir=%S/Inputs/resource_dir_amdgpu_triples %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CPU_NAME_SUBDIR %s

// Test that with a subarch triple, we can fall back to finding libraries
// in the generic (no-subarch) directory.
// RUN:   %clang -### --target=amdgpu9.0a-amd-amdhsa-llvm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_amdgpu_triples %s 2>&1 \
// RUN:   | FileCheck -check-prefix=FALLBACK-NO-SUBARCH %s

// Test that with a specific subarch triple, we can fall back to finding
// libraries in a compatible major version directory (e.g., amdgpu9.00 -> amdgpu9).
// RUN:   %clang -### --target=amdgpu9.00-amd-amdhsa-llvm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_amdgpu_major_version %s 2>&1 \
// RUN:   | FileCheck -check-prefix=FALLBACK-MAJOR-VERSION %s

// RUN:   %clang -### --target=amdgpu9-amd-amdhsa-llvm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_amdgpu_major_version %s 2>&1 \
// RUN:   | FileCheck -check-prefix=DIRECT-MAJOR-VERSION %s

// gfx90a is its own major subarch, so it does not fall back to the amdgpu9
// directory and no library is found.
// RUN:   not %clang -### --target=amdgpu9.0a-amd-amdhsa-llvm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_amdgpu_major_version %s 2>&1 \
// RUN:   | FileCheck -check-prefix=NO-FALLBACK-MAJOR-VERSION %s

// TOPLEVEL-SUBARCH: "-cc1" "-triple" "amdgpu10.30-amd-amdhsa-llvm" "{{.*}}/resource_dir_amdgpu_subarch_triples/lib/amdgpu10.30-amd-amdhsa-llvm/libclc.bc"

// CPU_NAME_SUBDIR: "-cc1" "-triple" "amdgpu9.0a-amd-amdhsa-llvm" "{{.*}}/resource_dir_amdgpu_triples/lib/amdgpu-amd-amdhsa-llvm/gfx90a/libclc.bc"

// FALLBACK-NO-SUBARCH: "-cc1" "-triple" "amdgpu9.0a-amd-amdhsa-llvm" "{{.*}}/resource_dir_amdgpu_triples/lib/amdgpu-amd-amdhsa-llvm/libclc.bc"

// FALLBACK-MAJOR-VERSION: "-cc1" "-triple" "amdgpu9.00-amd-amdhsa-llvm" "{{.*}}/resource_dir_amdgpu_major_version/lib/amdgpu9-amd-amdhsa-llvm/libclc.bc"

// DIRECT-MAJOR-VERSION: "-cc1" "-triple" "amdgpu9-amd-amdhsa-llvm" "{{.*}}/resource_dir_amdgpu_major_version/lib/amdgpu9-amd-amdhsa-llvm/libclc.bc"

// NO-FALLBACK-MAJOR-VERSION: no libclc library 'libclc.bc' found in the clang resource directory
