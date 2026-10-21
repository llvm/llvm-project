/// Tests for Intel GPU names in --offload-arch.

/// An Intel GPU name names a device job.
// RUN: %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-targets=spirv64-unknown-unknown --offload-arch=xe-bmg-g21 \
// RUN:   --offload-arch=xe-dg2 -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-SYCL %s
// CHK-SYCL-DAG: offload, "device-sycl (spirv64-unknown-unknown:xe-bmg-g21)"
// CHK-SYCL-DAG: offload, "device-sycl (spirv64-unknown-unknown:xe-dg2)"

/// A device name, its numeric name and an alias are one device job, labeled
/// with the device name.
// RUN: %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-targets=spirv64-unknown-unknown --offload-arch=xe-bmg-g21 \
// RUN:   --offload-arch=xe_20.1.0 -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-ONE-BMG %s
// CHK-ONE-BMG-NOT: offload, "device-sycl
// CHK-ONE-BMG: offload, "device-sycl (spirv64-unknown-unknown:xe-bmg-g21)"
// CHK-ONE-BMG-NOT: offload, "device-sycl
// RUN: %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-targets=spirv64-unknown-unknown --offload-arch=xe-arl-u \
// RUN:   --offload-arch=xe-mtl-u --offload-arch=xe_12.70 -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-ONE-MTL %s
// CHK-ONE-MTL-NOT: offload, "device-sycl
// CHK-ONE-MTL: offload, "device-sycl (spirv64-unknown-unknown:xe-mtl-u)"
// CHK-ONE-MTL-NOT: offload, "device-sycl
// RUN: %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-targets=spirv64-unknown-unknown --offload-arch=xe-tgllp -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-TGL %s
// CHK-TGL: offload, "device-sycl (spirv64-unknown-unknown:xe-tgl)"

/// Removing any spelling removes the device.
// RUN: %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-targets=spirv64-unknown-unknown --offload-arch=xe-bmg-g21 \
// RUN:   --offload-arch=generic --no-offload-arch=xe_20.1.0 -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-NO-ARCH %s
// RUN: %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-targets=spirv64-unknown-unknown --offload-arch=xe_20.1.0 \
// RUN:   --offload-arch=generic --no-offload-arch=xe-bmg-g21 -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-NO-ARCH %s
// CHK-NO-ARCH-NOT: bmg
// CHK-NO-ARCH: offload, "device-sycl (spirv64-unknown-unknown:generic)"
// CHK-NO-ARCH-NOT: bmg

/// -Xarch_ may name an Intel GPU by any spelling, including the one the user
/// gave --offload-arch.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-targets=spirv64-unknown-unknown --offload-arch=xe_20.1.0 \
// RUN:   -Xarch_xe_20.1.0 -DNUM -Xarch_xe-bmg-g21 -DBMG -Xarch_xe-pvc -DPVC -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-XARCH %s
// CHK-XARCH: "-fsycl-is-device"
// CHK-XARCH-SAME: "-D" "NUM"
// CHK-XARCH-SAME: "-D" "BMG"
// CHK-XARCH-NOT: "PVC"

/// The same holds for any offload language, e.g. OpenMP.
// RUN: %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fopenmp=libomp \
// RUN:   --offload-targets=spirv64-intel --offload-arch=xe-bmg-g21 \
// RUN:   --offload-arch=xe_20.1.0 -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-OPENMP %s
// CHK-OPENMP-NOT: offload, "device-openmp
// CHK-OPENMP: offload, "device-openmp (spirv64-intel:xe-bmg-g21)"
// CHK-OPENMP-NOT: offload, "device-openmp

/// Without --offload-targets, an Intel device or "generic" picks SPIR-V of the
/// host's width, and removing any spelling still removes the device.
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
// RUN:   --offload-arch=xe-pvc --offload-arch=xe_12.60.7 -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-NO-TARGETS-ONE %s
// CHK-NO-TARGETS-ONE-NOT: offload, "device-sycl
// CHK-NO-TARGETS-ONE: offload, "device-sycl (spirv64-unknown-unknown:xe-pvc)"
// CHK-NO-TARGETS-ONE-NOT: offload, "device-sycl
// RUN: %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-arch=xe_20.1.0 --offload-arch=xe-pvc --no-offload-arch=xe-bmg-g21 \
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
