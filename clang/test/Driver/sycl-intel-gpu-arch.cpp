/// Tests for Intel GPU names in --offload-arch.

/// An Intel GPU name names a device job.
// RUN: %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-targets=spirv64-unknown-unknown --offload-arch=xe-bmg-g21 \
// RUN:   --offload-arch=xe-dg2 -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-SYCL %s
// CHK-SYCL-DAG: offload, "device-sycl (spirv64-unknown-unknown:xe-bmg-g21)"
// CHK-SYCL-DAG: offload, "device-sycl (spirv64-unknown-unknown:xe-dg2)"

/// The same holds for any offload language, e.g. OpenMP.
// RUN: %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fopenmp=libomp \
// RUN:   --offload-targets=spirv64-intel --offload-arch=xe-bmg-g21 -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-OPENMP %s
// CHK-OPENMP: offload, "device-openmp (spirv64-intel:xe-bmg-g21)"
