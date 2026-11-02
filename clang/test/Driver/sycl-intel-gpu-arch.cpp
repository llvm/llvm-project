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
