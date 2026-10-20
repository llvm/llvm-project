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

/// Without --offload-targets, an Intel device or "generic" picks SPIR-V of the
/// host's width, and --no-offload-arch still removes the device.
// RUN: %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-arch=xe-pvc --offload-arch=xe-bmg-g21 --offload-arch=graniterapids \
// RUN:   --offload-arch=generic -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-NO-TARGETS %s
// CHK-NO-TARGETS-DAG: offload, "device-sycl (spirv64-unknown-unknown:generic)"
// CHK-NO-TARGETS-DAG: offload, "device-sycl (spirv64-unknown-unknown:graniterapids)"
// CHK-NO-TARGETS-DAG: offload, "device-sycl (spirv64-unknown-unknown:xe-bmg-g21)"
// CHK-NO-TARGETS-DAG: offload, "device-sycl (spirv64-unknown-unknown:xe-pvc)"
// RUN: %clang -ccc-print-phases --target=i386-unknown-linux-gnu -fsycl \
// RUN:   --offload-arch=xe-pvc -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-NO-TARGETS-32 %s
// CHK-NO-TARGETS-32: offload, "device-sycl (spirv32-unknown-unknown:xe-pvc)"
// RUN: %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-arch=xe-bmg-g21 --offload-arch=xe-pvc --no-offload-arch=xe-bmg-g21 \
// RUN:   -c %s 2>&1 | FileCheck -check-prefixes=CHK-NO-TARGETS-NO-ARCH %s
// CHK-NO-TARGETS-NO-ARCH-NOT: bmg
// CHK-NO-TARGETS-NO-ARCH: offload, "device-sycl (spirv64-unknown-unknown:xe-pvc)"
// CHK-NO-TARGETS-NO-ARCH-NOT: bmg

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
