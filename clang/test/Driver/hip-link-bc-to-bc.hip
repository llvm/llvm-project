// Check that clang unbundles the two bitcodes and links via llvm-link
// RUN: rm -rf %t && mkdir %t
// RUN: touch %t/bundle1.bc
// RUN: touch %t/bundle2.bc

// RUN: %clang -### --target=x86_64-unknown-linux-gnu --offload-arch=gfx906 --hip-link \
// RUN:   -nogpulib -nogpuinc -emit-llvm -fgpu-rdc --cuda-device-only \
// RUN:   --no-offload-new-driver %t/bundle1.bc %t/bundle2.bc \
// RUN:   2>&1 | FileCheck -check-prefix=BITCODE %s

// BITCODE: "{{.*}}clang-offload-bundler" "-type=bc" "-targets=host-x86_64-unknown-linux-gnu,hip-amdgcn-amd-amdhsa-gfx906" "-input={{.*}}bundle1.bc" "-output=[[B1HOST:.*\.bc]]" "-output=[[B1DEV1:.*\.bc]]" "-unbundle" "-allow-missing-bundles"
// BITCODE: "{{.*}}clang{{.*}}" "-o" "[[B1DEV2:.*bundle1-gfx906-.*\.bc]]" "-x" "ir" "[[B1DEV1]]"

// BITCODE: "{{.*}}clang-offload-bundler" "-type=bc" "-targets=host-x86_64-unknown-linux-gnu,hip-amdgcn-amd-amdhsa-gfx906" "-input={{.*}}bundle2.bc" "-output=[[B2HOST:.*\.bc]]" "-output=[[B2DEV1:.*\.bc]]" "-unbundle" "-allow-missing-bundles"
// BITCODE: "{{.*}}clang{{.*}}" "-o" "[[B2DEV2:.*bundle2-gfx906-.*\.bc]]" "-x" "ir" "[[B2DEV1]]"

// BITCODE: "{{.*}}llvm-link" "-o" "bundle1-hip-amdgcn-amd-amdhsa-gfx906.bc" "[[B1DEV2]]" "[[B2DEV2]]"

// Check that clang unbundles the bitcode and archive and links via llvm-link
// RUN: llvm-ar rc %t/libhipbundle.a
// RUN: touch %t/bundle.bc

// RUN: %clang -### --target=x86_64-unknown-linux-gnu --offload-arch=gfx906 --hip-link \
// RUN:   -nogpulib -nogpuinc -emit-llvm -fgpu-rdc --cuda-device-only \
// RUN:   --no-offload-new-driver %t/bundle.bc -L%t -lhipbundle \
// RUN:   2>&1 | FileCheck -check-prefix=ARCHIVE %s

// ARCHIVE: "{{.*}}clang-offload-bundler" "-type=bc" "-targets=host-x86_64-unknown-linux-gnu,hip-amdgcn-amd-amdhsa-gfx906" "-input={{.*}}bundle.bc" "-output=[[HOST:.*\.bc]]" "-output=[[DEV1:.*\.bc]]" "-unbundle" "-allow-missing-bundles"
// ARCHIVE: "{{.*}}clang{{.*}}" "-o" "[[DEV2:.*\.bc]]" "-x" "ir" "[[DEV1]]"

// ARCHIVE: "{{.*}}clang-offload-bundler" "-unbundle" "-type=a" "-input={{.*}}libhipbundle.a" "-targets=hip-amdgcn-amd-amdhsa-gfx906" "-output=[[AR:.*\.a]]" "-allow-missing-bundles" "-hip-openmp-compatible"

// ARCHIVE: "{{.*}}llvm-link" "-o" "bundle-hip-amdgcn-amd-amdhsa-gfx906.bc" "[[DEV2]]" "[[AR]]"
