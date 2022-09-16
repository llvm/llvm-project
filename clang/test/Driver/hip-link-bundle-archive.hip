// REQUIRES: x86-registered-target, amdgpu-registered-target


// Check clang unbundle the archive and link them by lld.

// RUN: rm -rf %t && mkdir %t
// RUN: touch %t/libhipBundled.a
// RUN: %clang -### --offload-arch=gfx906 --offload-arch=gfx1030 \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   -nogpulib %s -fgpu-rdc -L%t -lhipBundled \
// RUN:   2>&1 | FileCheck -check-prefix=GNU %s

// RUN: touch %t/hipBundled2.lib
// RUN: %clang -### --offload-arch=gfx906 --offload-arch=gfx1030 \
// RUN:   --target=x86_64-pc-windows-msvc \
// RUN:   -nogpulib %s -fgpu-rdc -L%t -lhipBundled2 \
// RUN:   2>&1 | FileCheck -check-prefix=MSVC %s

// GNU: "{{.*}}clang-offload-bundler" "-unbundle" "-type=a" "-input={{.*}}libhipBundled.a" "-targets=hip-amdgcn-amd-amdhsa-gfx1030" "-output=[[A1030:.*\.a]]" "-allow-missing-bundles"
// GNU: "{{.*}}lld{{.*}}" {{.*}}"-plugin-opt=mcpu=gfx1030" {{.*}} "[[A1030]]"
// GNU: "{{.*}}clang-offload-bundler" "-unbundle" "-type=a" "-input={{.*}}libhipBundled.a" "-targets=hip-amdgcn-amd-amdhsa-gfx906" "-output=[[A906:.*\.a]]" "-allow-missing-bundles"
// GNU: "{{.*}}lld{{.*}}" {{.*}}"-plugin-opt=mcpu=gfx906" {{.*}} "[[A906]]"
// GNU: "{{.*}}ld{{.*}}" {{.*}}"-o" "a.out" {{.*}}"-lhipBundled"

// MSVC: "{{.*}}clang-offload-bundler" "-unbundle" "-type=a" "-input={{.*}}hipBundled2.lib" "-targets=hip-amdgcn-amd-amdhsa-gfx1030" "-output=[[A1030:.*\.a]]" "-allow-missing-bundles"
// MSVC: "{{.*}}lld{{.*}}" {{.*}}"-plugin-opt=mcpu=gfx1030" {{.*}} "[[A1030]]"
// MSVC: "{{.*}}clang-offload-bundler" "-unbundle" "-type=a" "-input={{.*}}hipBundled2.lib" "-targets=hip-amdgcn-amd-amdhsa-gfx906" "-output=[[A906:.*\.a]]" "-allow-missing-bundles"
// MSVC: "{{.*}}lld{{.*}}" {{.*}}"-plugin-opt=mcpu=gfx906" {{.*}} "[[A906]]"
// MSVC: "{{.*}}link{{.*}}" {{.*}}"-out:a.exe" {{.*}}"hipBundled2.lib"
