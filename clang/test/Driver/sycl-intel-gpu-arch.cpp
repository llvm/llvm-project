/// Tests for Intel GPU names in --offload-arch.

/// A device name and its numeric name are one architecture, spelled with the
/// device name.
// RUN: %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-targets=spirv64-unknown-unknown --offload-arch=xe-bmg-g21 \
// RUN:   --offload-arch=xe_20.1.0 -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-SPELLINGS %s
// CHK-SPELLINGS-NOT: offload, "device-sycl
// CHK-SPELLINGS: offload, "device-sycl (spirv64-unknown-unknown:xe-bmg-g21)"
// CHK-SPELLINGS-NOT: offload, "device-sycl

/// Removing either spelling removes the device.
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

/// An Intel GPU name takes no target-ID features, so a suffix is not dropped to
/// turn the name into a device.
// RUN: %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-targets=spirv64-unknown-unknown --offload-arch=xe-pvc:foo -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-SUFFIX %s
// CHK-SUFFIX: offload, "device-sycl (spirv64-unknown-unknown:xe-pvc:foo)"

/// -Xarch_ may name an Intel GPU by either spelling.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-targets=spirv64-unknown-unknown --offload-arch=xe_20.1.0 \
// RUN:   -Xarch_xe-bmg-g21 -DBMG -Xarch_xe-pvc -DPVC -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-XARCH %s
// CHK-XARCH: "-fsycl-is-device"
// CHK-XARCH-SAME: "-D" "BMG"
// CHK-XARCH-NOT: "PVC"

/// A static library's device code is requested by the device name.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-targets=spirv64-unknown-unknown --offload-arch=xe_20.1.0 %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-EXTRACT %s
// CHK-EXTRACT: clang-linker-wrapper{{.*}}"--should-extract=xe-bmg-g21"

/// The same holds for any offload language, e.g. OpenMP.
// RUN: %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fopenmp=libomp \
// RUN:   --offload-targets=spirv64-intel --offload-arch=xe-bmg-g21 \
// RUN:   --offload-arch=xe_20.1.0 -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-OPENMP %s
// CHK-OPENMP-NOT: offload, "device-openmp
// CHK-OPENMP: offload, "device-openmp (spirv64-intel:xe-bmg-g21)"
// CHK-OPENMP-NOT: offload, "device-openmp
