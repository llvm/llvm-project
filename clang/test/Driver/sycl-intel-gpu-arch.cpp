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

/// Without --offload-targets, an Intel device, an IGCA target or "generic" picks
/// SPIR-V of the host's width.
// RUN: %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-arch=xe-pvc --offload-arch=xe-bmg-g21 --offload-arch=graniterapids \
// RUN:   --offload-arch=generic --offload-arch=igca_20ca -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-NO-TARGETS %s
// CHK-NO-TARGETS-DAG: offload, "device-sycl (spirv64-unknown-unknown:generic)"
// CHK-NO-TARGETS-DAG: offload, "device-sycl (spirv64-unknown-unknown:graniterapids)"
// CHK-NO-TARGETS-DAG: offload, "device-sycl (spirv64-unknown-unknown:igca_20ca)"
// CHK-NO-TARGETS-DAG: offload, "device-sycl (spirv64-unknown-unknown:xe-bmg-g21)"
// CHK-NO-TARGETS-DAG: offload, "device-sycl (spirv64-unknown-unknown:xe-pvc)"
// RUN: %clang -ccc-print-phases --target=i386-unknown-linux-gnu -fsycl \
// RUN:   --offload-arch=xe-pvc --offload-arch=igca_20ca -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-NO-TARGETS-32 %s
// CHK-NO-TARGETS-32-DAG: offload, "device-sycl (spirv32-unknown-unknown:xe-pvc)"
// CHK-NO-TARGETS-32-DAG: offload, "device-sycl (spirv32-unknown-unknown:igca_20ca)"
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

/// On a SPIR-V target, a SYCL device must be an Intel GPU or CPU, or "generic".
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
/// A numeric name names no device unless the list has its GPU IP version.
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-targets=spirv64-unknown-unknown --offload-arch=xe_40.11.0 -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-BAD-ARCH-UNLISTED %s
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-arch=xe_40.11.0 -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-BAD-ARCH-UNLISTED %s
// CHK-BAD-ARCH-UNLISTED: error: unsupported {{SYCL|offload}} gpu architecture: xe_40.11.0
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
// RUN:   -c %s 2>&1 | FileCheck -check-prefixes=CHK-GOOD-ARCH %s
// CHK-GOOD-ARCH: offload, "device-sycl (spirv64-unknown-unknown:graniterapids)"
