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

/// It has no effect for an IGCA target given directly, which is compiled just
/// in time, also when one of its Intel GPUs is given as well.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-arch=igca_30r --no-offload-embed-virtual-arch -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=DIRECT %s
// DIRECT-NOT: warning:
// DIRECT: "--image=file={{[^"]*}},triple=spirv64-unknown-unknown,arch=igca_30r,kind=sycl"{{$}}
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

/// It has no effect without --offload-arch.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --no-offload-embed-virtual-arch -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=GENERIC %s
// GENERIC-NOT: warning:
// GENERIC: "--image=file={{[^"]*}},triple=spirv64-unknown-unknown,arch=generic,kind=sycl"{{$}}
