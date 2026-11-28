/// Tests for --[no-]offload-embed-virtual-arch with SYCL.

/// By default, the SPIR-V is kept.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-arch=xe-mtl-u -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=EMBED %s
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-arch=xe-mtl-u --no-offload-embed-virtual-arch \
// RUN:   --offload-embed-virtual-arch -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=EMBED %s
// EMBED: "--image=file={{[^"]*}},triple=spirv64-unknown-unknown,arch=igca_30r,kind=sycl,code=xe-mtl-u"{{$}}

/// --no-offload-embed-virtual-arch is recorded for a device image compiled
/// ahead of time.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-arch=xe-mtl-u --no-offload-embed-virtual-arch -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=NO-EMBED %s
// NO-EMBED: "--image=file={{[^"]*}},triple=spirv64-unknown-unknown,arch=igca_30r,kind=sycl,code=xe-mtl-u,no-embed-virtual-arch=1"{{$}}

/// Where the option can't do anything, it is reported as unused.
// UNUSED: warning: argument unused during compilation: '--no-offload-embed-virtual-arch'

/// It has no effect for an IGCA target given in --offload-arch, so it is
/// reported as unused.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-arch=igca_30r --no-offload-embed-virtual-arch -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=DIRECT,UNUSED %s
// DIRECT: "--image=file={{[^"]*}},triple=spirv64-unknown-unknown,arch=igca_30r,kind=sycl"{{$}}

/// It still has no effect when one of the IGCA target's Intel GPUs is also
/// given.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-arch=xe-mtl-u,igca_30r --no-offload-embed-virtual-arch \
// RUN:   -c %s 2>&1 | FileCheck -check-prefix=EMBED %s

/// With -Xarch_, it applies only to the device compile for that IGCA target.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-arch=xe-mtl-u,xe-bmg-g21 \
// RUN:   -Xarch_igca_40r --no-offload-embed-virtual-arch -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=XARCH %s
// XARCH-DAG: "--image=file={{[^"]*}},triple=spirv64-unknown-unknown,arch=igca_30r,kind=sycl,code=xe-mtl-u"
// XARCH-DAG: "--image=file={{[^"]*}},triple=spirv64-unknown-unknown,arch=igca_40r,kind=sycl,code=xe-bmg-g21,no-embed-virtual-arch=1"

/// It has no effect without --offload-arch, so it is reported as unused.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --no-offload-embed-virtual-arch -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=GENERIC,UNUSED %s
// GENERIC: "--image=file={{[^"]*}},triple=spirv64-unknown-unknown,arch=generic,kind=sycl"{{$}}

/// It is SYCL only, so it is reported as unused for OpenMP.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp \
// RUN:   --offload-arch=sm_70 -nogpulib --no-offload-embed-virtual-arch -c %s \
// RUN:   2>&1 | FileCheck -check-prefix=UNUSED %s

