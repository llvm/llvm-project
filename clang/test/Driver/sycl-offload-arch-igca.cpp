/// Tests for Intel IGCA targets in --offload-arch.

// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-targets=spirv64-unknown-unknown --offload-arch=igca_20ca \
// RUN:   -c %s 2>&1 | FileCheck %s
// CHECK: llvm-offload-binary{{.*}} "--image=file={{.*}}.bc,triple=spirv64-unknown-unknown,arch=igca_20ca,kind=sycl"

/// Reject IGCA targets not listed in IntelGPUTargetParser.def:
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-targets=spirv64-unknown-unknown --offload-arch=igca_42a \
// RUN:   -c %s 2>&1 | FileCheck -check-prefix=CHK-INVALID %s
// CHK-INVALID: error: unsupported IGCA gpu architecture: igca_42a
// CHK-INVALID-NOT: llvm-offload-binary

/// IGCA targets should not accept feature suffixes, so a valid target followed
/// by a feature suffix should be rejected: 
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-targets=spirv64-unknown-unknown --offload-arch=igca_20ca:foo \
// RUN:   -c %s 2>&1 | FileCheck -check-prefix=CHK-FEATURE %s
// CHK-FEATURE: error: unsupported IGCA gpu architecture: igca_20ca:foo
// CHK-FEATURE-NOT: llvm-offload-binary

/// A valid IGCA target is still rejected on NVPTX and AMDGPU device triples,
/// with the existing CUDA/HIP diagnostic:
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-targets=nvptx64-nvidia-cuda --offload-arch=igca_20ca \
// RUN:   -c %s 2>&1 | FileCheck -check-prefix=CHK-NVPTX %s
// CHK-NVPTX: error: unsupported CUDA gpu architecture: igca_20ca
// CHK-NVPTX-NOT: llvm-offload-binary

// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-targets=amdgcn-amd-amdhsa --offload-arch=igca_20ca \
// RUN:   -c %s 2>&1 | FileCheck -check-prefix=CHK-AMDGPU %s
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-targets=spirv64-amd-amdhsa --offload-arch=igca_20ca \
// RUN:   -c %s 2>&1 | FileCheck -check-prefix=CHK-AMDGPU %s
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-targets=spirv64-unknown-amdhsa --offload-arch=igca_20ca \
// RUN:   -c %s 2>&1 | FileCheck -check-prefix=CHK-AMDGPU %s
// CHK-AMDGPU: error: unsupported HIP gpu architecture: igca_20ca
// CHK-AMDGPU-NOT: llvm-offload-binary
