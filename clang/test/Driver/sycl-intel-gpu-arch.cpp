/// Tests for Intel GPU names in --offload-arch.

/// An Intel GPU name names a device job.
// RUN: %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-targets=spirv64-unknown-unknown --offload-arch=xe-bmg-g21 \
// RUN:   --offload-arch=xe-dg2 -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-SYCL %s
// CHK-SYCL-DAG: offload, "device-sycl (spirv64-unknown-unknown:xe-bmg-g21)"
// CHK-SYCL-DAG: offload, "device-sycl (spirv64-unknown-unknown:xe-dg2)"

/// A device name, an alias and a numeric name are kept as given.
// RUN: %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-targets=spirv64-unknown-unknown --offload-arch=xe-mtl-u \
// RUN:   --offload-arch=xe-arl-u --offload-arch=xe_12.70.0 -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-AS-GIVEN %s
// CHK-AS-GIVEN-DAG: offload, "device-sycl (spirv64-unknown-unknown:xe-mtl-u)"
// CHK-AS-GIVEN-DAG: offload, "device-sycl (spirv64-unknown-unknown:xe-arl-u)"
// CHK-AS-GIVEN-DAG: offload, "device-sycl (spirv64-unknown-unknown:xe_12.70.0)"

/// Without --offload-targets, an Intel device (by name or numeric name), an IGCA
/// target or "generic" picks SPIR-V of the host's width.
// RUN: %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-arch=xe-pvc --offload-arch=xe-bmg-g21 --offload-arch=graniterapids \
// RUN:   --offload-arch=generic --offload-arch=igca_20ca --offload-arch=xe_12.60.7 \
// RUN:   -c %s 2>&1 | FileCheck -check-prefixes=CHK-NO-TARGETS %s
// CHK-NO-TARGETS-DAG: offload, "device-sycl (spirv64-unknown-unknown:generic)"
// CHK-NO-TARGETS-DAG: offload, "device-sycl (spirv64-unknown-unknown:graniterapids)"
// CHK-NO-TARGETS-DAG: offload, "device-sycl (spirv64-unknown-unknown:igca_20ca)"
// CHK-NO-TARGETS-DAG: offload, "device-sycl (spirv64-unknown-unknown:xe-bmg-g21)"
// CHK-NO-TARGETS-DAG: offload, "device-sycl (spirv64-unknown-unknown:xe-pvc)"
// CHK-NO-TARGETS-DAG: offload, "device-sycl (spirv64-unknown-unknown:xe_12.60.7)"
// RUN: %clang -ccc-print-phases --target=i386-unknown-linux-gnu -fsycl \
// RUN:   --offload-arch=xe-pvc --offload-arch=igca_20ca -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-NO-TARGETS-32 %s
// CHK-NO-TARGETS-32-DAG: offload, "device-sycl (spirv32-unknown-unknown:xe-pvc)"
// CHK-NO-TARGETS-32-DAG: offload, "device-sycl (spirv32-unknown-unknown:igca_20ca)"
/// Names are kept as given, so two aliases of one device make two device jobs,
/// and --no-offload-arch removes a name.
// RUN: %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-arch=xe-mtl-u --offload-arch=xe-arl-u --offload-arch=xe-pvc \
// RUN:   --no-offload-arch=xe-pvc -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-NO-TARGETS-AS-GIVEN %s
// CHK-NO-TARGETS-AS-GIVEN-NOT: xe-pvc
// CHK-NO-TARGETS-AS-GIVEN-DAG: offload, "device-sycl (spirv64-unknown-unknown:xe-mtl-u)"
// CHK-NO-TARGETS-AS-GIVEN-DAG: offload, "device-sycl (spirv64-unknown-unknown:xe-arl-u)"
// CHK-NO-TARGETS-AS-GIVEN-NOT: xe-pvc

/// As for CUDA and HIP, -foffload-via-llvm gives the inferred target the LLVM
/// environment.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl -foffload-via-llvm \
// RUN:   --offload-arch=xe-pvc -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-VIA-LLVM %s
// CHK-VIA-LLVM: "-cc1" "-triple" "spirv64-unknown-unknown-llvm"

/// Only SYCL infers a target from an Intel name.
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -x cuda -nocudainc \
// RUN:   -nocudalib --offload-arch=xe-pvc -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-CUDA %s
// CHK-CUDA: error: unsupported CUDA gpu architecture: xe-pvc
