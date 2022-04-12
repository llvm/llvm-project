// REQUIRES: x86-registered-target, amdgpu-registered-target

// RUN: touch %T/libhipBundled.a

// Check clang unbundle the archive and link them by lld.

// RUN: %clang -### --offload-arch=gfx906 --offload-arch=gfx1030 \
// RUN:   -nogpulib %s -fgpu-rdc -L%T -lhipBundled \
// RUN:   2>&1 | FileCheck -check-prefix=CHECK %s

// CHECK: "{{.*}}clang-offload-bundler" "-unbundle" "-type=a" "-input={{.*}}libhipBundled.a" "-targets=hip-amdgcn-amd-amdhsa-gfx1030" "-output=[[A1030:.*\.a]]" "-allow-missing-bundles"
// CHECK: "{{.*}}lld{{.*}}" {{.*}}"-plugin-opt=mcpu=gfx1030" {{.*}} "[[A1030]]"
// CHECK: "{{.*}}clang-offload-bundler" "-unbundle" "-type=a" "-input={{.*}}libhipBundled.a" "-targets=hip-amdgcn-amd-amdhsa-gfx906" "-output=[[A906:.*\.a]]" "-allow-missing-bundles"
// CHECK: "{{.*}}lld{{.*}}" {{.*}}"-plugin-opt=mcpu=gfx906" {{.*}} "[[A906]]"
