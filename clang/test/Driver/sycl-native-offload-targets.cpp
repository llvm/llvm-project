// Needs chmod
// UNSUPPORTED: system-windows
// XFAIL: target={{.*}}-zos{{.*}}

// RUN: mkdir -p %t
// RUN: cp %S/Inputs/offload-arch/offload_arch_sm_70_xe_pvc_by_vendor %t/
// RUN: chmod +x %t/offload_arch_sm_70_xe_pvc_by_vendor

/// With a SPIR-V target, --offload-arch=native picks the system's Intel GPUs.
// RUN: %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-targets=spirv64-unknown-unknown --offload-arch=native \
// RUN:   --offload-arch-tool=%t/offload_arch_sm_70_xe_pvc_by_vendor -c %s 2>&1 \
// RUN:   | FileCheck %s
// CHECK-NOT: sm_70
// CHECK: offload, "device-sycl (spirv64-unknown-unknown:xe-pvc)"
// CHECK-NOT: sm_70
