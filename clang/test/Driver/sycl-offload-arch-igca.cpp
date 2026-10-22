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

/// FIXME(IGCA): Enable this when OffloadArchToTriple gives IGCA targets a triple.
// TODO-RUN %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// TODO-RUN   --offload-arch=igca_20ca -c %s 2>&1 | FileCheck -check-prefix=CHK-INFER %s
// CHK-INFER-NOT: error:
// CHK-INFER: llvm-offload-binary{{.*}} "--image=file={{.*}}.bc,triple=spirv64{{[^,]*}},arch=igca_20ca,kind=sycl"
