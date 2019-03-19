// RUN: %clang -### --sycl -c %s 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang -### --sycl -fno-sycl-use-bitcode -c %s 2>&1 | FileCheck %s --check-prefix=NO-BITCODE
// RUN: %clang -### -target spir64-unknown-linux-sycldevice -c -emit-llvm %s 2>&1 | FileCheck %s --check-prefix=TARGET
// RUN: %clang -### --sycl -c -emit-llvm %s 2>&1 | FileCheck %s --check-prefix=COMBINED

// DEFAULT: "-triple" "spir64-unknown-{{.*}}-sycldevice"{{.*}} "-fsycl-is-device"{{.*}} "-emit-llvm-bc"
// DEFAULT-NOT: "{{.*}}llvm-spirv"{{.*}} "-spirv-no-deref-attr"
// NO-BITCODE: "-triple" "spir64-unknown-{{.*}}-sycldevice"{{.*}} "-fsycl-is-device"{{.*}} "-emit-llvm-bc"
// NO-BITCODE: "{{.*}}llvm-spirv"{{.*}} "-spirv-no-deref-attr"
// TARGET: "-triple" "spir64-unknown-linux-sycldevice"{{.*}} "-fsycl-is-device"{{.*}} "-emit-llvm-bc"
// COMBINED: "-triple" "spir64-unknown-{{.*}}-sycldevice"{{.*}} "-fsycl-is-device"{{.*}} "-emit-llvm-bc"
