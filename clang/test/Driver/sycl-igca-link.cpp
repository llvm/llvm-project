/// SYCL device code for Intel GPUs is compiled once per IGCA target. At link
/// time, SYCL device code for the same IGCA target and devices is linked
/// together, and the devices are passed on with --code.

// REQUIRES: x86-registered-target
// UNSUPPORTED: system-windows

// RUN: %clang -fsycl --target=x86_64-unknown-linux-gnu \
// RUN:   --offload-arch=xe-mtl-u,xe-mtl-h -c %s -o %t-uh-a.o
// RUN: %clang -fsycl --target=x86_64-unknown-linux-gnu \
// RUN:   --offload-arch=xe-mtl-h,xe-mtl-u -c %s -o %t-uh-b.o
// RUN: %clang -fsycl --target=x86_64-unknown-linux-gnu \
// RUN:   --offload-arch=xe-mtl-u -c %s -o %t-u.o

/// xe-mtl-u and xe-mtl-h share igca_30r, so the SYCL device code of both
/// objects is linked together, with both devices passed on with --code.
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t-uh-a.o %t-uh-b.o -o a.out 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SAME-DEVICES
// SAME-DEVICES:     clang{{.*}} -march=igca_30r {{.*}}.o {{.*}}.o --sycl-link -Xlinker --code=xe-mtl-h -Xlinker --code=xe-mtl-u{{$}}
// SAME-DEVICES-NOT: --sycl-link

/// An object compiled for xe-mtl-u only has a different list of devices, so its
/// SYCL device code is linked separately.
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t-u.o %t-uh-a.o -o a.out 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DIFFERENT-DEVICES
// DIFFERENT-DEVICES-DAG: clang{{.*}} -march=igca_30r {{.*}}.o --sycl-link -Xlinker --code=xe-mtl-u{{$}}
// DIFFERENT-DEVICES-DAG: clang{{.*}} -march=igca_30r {{.*}}.o --sycl-link -Xlinker --code=xe-mtl-h -Xlinker --code=xe-mtl-u{{$}}
