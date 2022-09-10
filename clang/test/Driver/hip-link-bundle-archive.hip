// REQUIRES: x86-registered-target, amdgpu-registered-target

// Check clang unbundle the archive and link them by lld.

// RUN: rm -rf %t && mkdir %t
// RUN: touch %t/dummy.bc
// RUN: llvm-ar cr %t/libhipBundled.a %t/dummy.bc
// RUN: %clang -### --offload-arch=gfx906 --offload-arch=gfx1030 \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   -nogpulib %s -fgpu-rdc -L%t -lhipBundled \
// RUN:   2>&1 | FileCheck -check-prefixes=GNU,GNU1,GNU-L %s

// RUN: %clang -### --offload-arch=gfx906 --offload-arch=gfx1030 \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   -nogpulib %s -fgpu-rdc -L%t -l:libhipBundled.a \
// RUN:   2>&1 | FileCheck -check-prefixes=GNU,GNU1,GNU-LA %s

// RUN: %clang -### --offload-arch=gfx906 --offload-arch=gfx1030 \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   -nogpulib %s -fgpu-rdc %t/libhipBundled.a \
// RUN:   2>&1 | FileCheck -check-prefixes=GNU,GNU1,GNU-A %s

// RUN: llvm-ar cr %t/libhipBundled.a.5.2 %t/dummy.bc
// RUN: %clang -### --offload-arch=gfx906 --offload-arch=gfx1030 \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   -nogpulib %s -fgpu-rdc %t/libhipBundled.a.5.2 \
// RUN:   2>&1 | FileCheck -check-prefixes=GNU,GNU2,GNU-A %s

// Check if a file is not an archive, it is not unbundled.

// RUN: touch %t/libNonArchive.a
// RUN: %clang -### --offload-arch=gfx906 --offload-arch=gfx1030 \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   -nogpulib %s -fgpu-rdc -L%t -lNonArchive \
// RUN:   2>&1 | FileCheck -check-prefixes=NONARCHIVE %s
// RUN: %clang -### --offload-arch=gfx906 --offload-arch=gfx1030 \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   -nogpulib %s -fgpu-rdc -L%t -l:libNonArchive.a \
// RUN:   2>&1 | FileCheck -check-prefixes=NONARCHIVE %s
// RUN: %clang -### --offload-arch=gfx906 --offload-arch=gfx1030 \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   -nogpulib %s -fgpu-rdc -L%t libNonArchive.a \
// RUN:   2>&1 | FileCheck -check-prefixes=NONARCHIVE %s

// Check unbundling archive for MSVC.

// RUN: llvm-ar cr %t/hipBundled2.lib %t/dummy.bc
// RUN: %clang -### --offload-arch=gfx906 --offload-arch=gfx1030 \
// RUN:   --target=x86_64-pc-windows-msvc \
// RUN:   -nogpulib %s -fgpu-rdc -L%t -lhipBundled2 \
// RUN:   2>&1 | FileCheck -check-prefix=MSVC %s

// RUN: %clang -### --offload-arch=gfx906 --offload-arch=gfx1030 \
// RUN:   --target=x86_64-pc-windows-msvc \
// RUN:   -nogpulib %s -fgpu-rdc -L%t -l:hipBundled2.lib \
// RUN:   2>&1 | FileCheck -check-prefix=MSVC %s

// RUN: %clang -### --offload-arch=gfx906 --offload-arch=gfx1030 \
// RUN:   --target=x86_64-pc-windows-msvc \
// RUN:   -nogpulib %s -fgpu-rdc %t/hipBundled2.lib \
// RUN:   2>&1 | FileCheck -check-prefix=MSVC %s

// GNU1: "{{.*}}clang-offload-bundler" "-unbundle" "-type=a" "-input={{.*}}[[LIB:libhipBundled\.a]]" "-targets=hip-amdgcn-amd-amdhsa-gfx1030" "-output=[[A1030:.*\.a]]" "-allow-missing-bundles"
// GNU2: "{{.*}}clang-offload-bundler" "-unbundle" "-type=a" "-input={{.*}}[[LIB:libhipBundled\.a\.5\.2]]" "-targets=hip-amdgcn-amd-amdhsa-gfx1030" "-output=[[A1030:.*\.a]]" "-allow-missing-bundles"
// GNU: "{{.*}}lld{{.*}}" {{.*}}"-plugin-opt=mcpu=gfx1030" {{.*}} "[[A1030]]"
// GNU: "{{.*}}clang-offload-bundler" "-unbundle" "-type=a" "-input={{.*}}[[LIB]]" "-targets=hip-amdgcn-amd-amdhsa-gfx906" "-output=[[A906:.*\.a]]" "-allow-missing-bundles"
// GNU: "{{.*}}lld{{.*}}" {{.*}}"-plugin-opt=mcpu=gfx906" {{.*}} "[[A906]]"
// GNU-L: "{{.*}}ld{{.*}}" {{.*}}"-o" "a.out" {{.*}}"-lhipBundled"
// GNU-LA: "{{.*}}ld{{.*}}" {{.*}}"-o" "a.out" {{.*}}"-l:libhipBundled.a"
// GNU-A: "{{.*}}ld{{.*}}" {{.*}}"-o" "a.out" "{{.*}}[[LIB]]"
// NONARCHIVE-NOT: "{{.*}}clang-offload-bundler" "-unbundle" "-type=a" "-input={{.*libNonArchive\.a}}"

// MSVC: "{{.*}}clang-offload-bundler" "-unbundle" "-type=a" "-input={{.*}}hipBundled2.lib" "-targets=hip-amdgcn-amd-amdhsa-gfx1030" "-output=[[A1030:.*\.a]]" "-allow-missing-bundles"
// MSVC: "{{.*}}lld{{.*}}" {{.*}}"-plugin-opt=mcpu=gfx1030" {{.*}} "[[A1030]]"
// MSVC: "{{.*}}clang-offload-bundler" "-unbundle" "-type=a" "-input={{.*}}hipBundled2.lib" "-targets=hip-amdgcn-amd-amdhsa-gfx906" "-output=[[A906:.*\.a]]" "-allow-missing-bundles"
// MSVC: "{{.*}}lld{{.*}}" {{.*}}"-plugin-opt=mcpu=gfx906" {{.*}} "[[A906]]"
// MSVC: "{{.*}}link{{.*}}" {{.*}}"-out:a.exe" {{.*}}hipBundled2.lib"
