/// Device code for Intel GPUs is compiled once per IGCA target. At link time,
/// device code for the same IGCA target and devices is linked together, and
/// compiled ahead of time for those devices.

// REQUIRES: x86-registered-target
// UNSUPPORTED: system-windows

// RUN: %clang -fsycl --target=x86_64-unknown-linux-gnu \
// RUN:   --offload-arch=xe-mtl-u,xe-mtl-h -c %s -o %t-uh-a.o
// RUN: %clang -fsycl --target=x86_64-unknown-linux-gnu \
// RUN:   --offload-arch=xe-mtl-h,xe-mtl-u -c %s -o %t-uh-b.o
// RUN: %clang -fsycl --target=x86_64-unknown-linux-gnu \
// RUN:   --offload-arch=xe-mtl-u -c %s -o %t-u.o

/// xe-mtl-u and xe-mtl-h share igca_30r, so both objects have one device image
/// for igca_30r with the same devices, and their device code is linked together
/// for both devices.
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t-uh-a.o %t-uh-b.o -o a.out 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SAME-DEVICES
// SAME-DEVICES:     clang{{.*}} -march=igca_30r {{.*}}.o {{.*}}.o --sycl-link -Xlinker --device=xe-mtl-h -Xlinker --device=xe-mtl-u{{$}}
// SAME-DEVICES-NOT: --sycl-link

/// An object compiled for xe-mtl-u only has a different list of devices, so its
/// device code is linked separately.
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t-u.o %t-uh-a.o -o a.out 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DIFFERENT-DEVICES
// DIFFERENT-DEVICES-DAG: clang{{.*}} -march=igca_30r {{.*}}.o --sycl-link -Xlinker --device=xe-mtl-u{{$}}
// DIFFERENT-DEVICES-DAG: clang{{.*}} -march=igca_30r {{.*}}.o --sycl-link -Xlinker --device=xe-mtl-h -Xlinker --device=xe-mtl-u{{$}}
