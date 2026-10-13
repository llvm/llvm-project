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

/// Without --offload-targets, an Intel device or "generic" picks SPIR-V of the
/// host's width, and removing either spelling still removes the device.
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

/// On a SPIR-V target, a SYCL device must be an Intel GPU or CPU, "generic", or
/// the numeric name of a GPU the list does not know yet.
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-targets=spirv64-unknown-unknown --offload-arch=xe-pcv -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-BAD-ARCH %s
// CHK-BAD-ARCH: error: unsupported SYCL gpu architecture: xe-pcv
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-targets=spirv64-unknown-unknown --offload-arch=bmg_g21 -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-BAD-ARCH-OLD %s
// CHK-BAD-ARCH-OLD: error: unsupported SYCL gpu architecture: bmg_g21
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-targets=spirv64-unknown-unknown --offload-arch=xe_12.60. -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-BAD-ARCH-NUM %s
// CHK-BAD-ARCH-NUM: error: unsupported SYCL gpu architecture: xe_12.60.
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-targets=spirv64-unknown-unknown --offload-arch=xe-pvc:garbage -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-BAD-ARCH-SUFFIX %s
// CHK-BAD-ARCH-SUFFIX: error: unsupported SYCL gpu architecture: xe-pvc:garbage
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-targets=spirv64-unknown-unknown --offload-arch=generic:foo -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-BAD-GENERIC %s
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-arch=generic:foo -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-BAD-GENERIC %s
// CHK-BAD-GENERIC: error: unsupported {{SYCL|offload}} gpu architecture: generic:foo
/// As for CUDA and HIP, --no-offload-arch is checked too when the target is
/// given.
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-targets=spirv64-unknown-unknown --offload-arch=xe-pvc \
// RUN:   --no-offload-arch=foo -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-BAD-NO-ARCH %s
// CHK-BAD-NO-ARCH: error: unsupported SYCL gpu architecture: foo
/// Only a plain SPIR-V target is checked.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-targets=spirv64-intel --offload-arch=foo -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-NOT-CHECKED %s
// CHK-NOT-CHECKED-NOT: error:
// RUN: %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-targets=spirv64-unknown-unknown --offload-arch=graniterapids \
// RUN:   --offload-arch=xe_40.11.0 -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-GOOD-ARCH %s
// RUN: %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-arch=graniterapids --offload-arch=xe_40.11.0 -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-GOOD-ARCH %s
// CHK-GOOD-ARCH-DAG: offload, "device-sycl (spirv64-unknown-unknown:graniterapids)"
// CHK-GOOD-ARCH-DAG: offload, "device-sycl (spirv64-unknown-unknown:xe_40.11.0)"
