// RUN: %clang -### --target=amdgcn-amd-amdhsa-opencl -x cl -c -emit-llvm -mcpu=fiji -O0 %s 2>&1 | FileCheck -check-prefix=CHECK_O0 %s
// RUN: %clang -### --target=amdgcn-amd-amdhsa-opencl -x cl -c -emit-llvm -mcpu=fiji -O1 %s 2>&1 | FileCheck -check-prefix=CHECK_O1 %s
// RUN: %clang -### --target=amdgcn-amd-amdhsa-opencl -x cl -c -emit-llvm -mcpu=fiji -O2 %s 2>&1 | FileCheck -check-prefix=CHECK_O2 %s
// RUN: %clang -### --target=amdgcn-amd-amdhsa-opencl -x cl -c -emit-llvm -mcpu=fiji -O3 %s 2>&1 | FileCheck -check-prefix=CHECK_O3 %s
// RUN: %clang -### --target=amdgcn-amd-amdhsa-opencl -x cl -c -emit-llvm -mcpu=fiji -O4 %s 2>&1 | FileCheck -check-prefix=CHECK_O4 %s
// RUN: %clang -### --target=amdgcn-amd-amdhsa-opencl -x cl -c -emit-llvm -mcpu=fiji -O5 %s 2>&1 | FileCheck -check-prefix=CHECK_O5 %s
// RUN: %clang -### --target=amdgcn-amd-amdhsa-opencl -x cl -c -emit-llvm -mcpu=fiji -Og %s 2>&1 | FileCheck -check-prefix=CHECK_Og %s
// RUN: %clang -### --target=amdgcn-amd-amdhsa-opencl -x cl -c -emit-llvm -mcpu=fiji -Ofast %s 2>&1 | FileCheck -check-prefix=CHECK_Ofast %s
// RUN: %clang -### --target=amdgcn-amd-amdhsa-opencl -x cl -c -emit-llvm -mcpu=fiji %s 2>&1 | FileCheck -check-prefix=CHECK_O_DEFAULT %s

// Check default include file is not included for preprocessor output.

// RUN: %clang -### --target=amdgcn-amd-amdhsa-opencl -x cl -c -emit-llvm -mcpu=fiji %s 2>&1 | FileCheck -check-prefix=CHK-INC %s
// RUN: %clang -### --target=amdgcn-amd-amdhsa-opencl -x cl -c -emit-llvm -mcpu=fiji -save-temps %s 2>&1 | FileCheck -check-prefix=CHK-INC %s

// CHECK_O0: "-cc1"{{.*}} "-O0"
// CHECK_O1: "-cc1"{{.*}} "-O1"
// CHECK_O2: "-cc1"{{.*}} "-O2"
// CHECK_O3: "-cc1"{{.*}} "-O3"
// CHECK_O4: "-cc1"{{.*}} "-O3"
// CHECK_O5: "-cc1"{{.*}} "-O5"
// CHECK_Og: "-cc1"{{.*}} "-Og"
// CHECK_Ofast: "-cc1"{{.*}} "-Ofast"
// CHECK_O_DEFAULT: "-cc1"{{.*}} "-O3"

// CHK-INC: "-cc1" {{.*}}"-finclude-default-header" "-fdeclare-opencl-builtins" {{.*}}"-x" "cl"
// CHK-INC-NOT: "-cc1" {{.*}}"-finclude-default-header" "-fdeclare-opencl-builtins" {{.*}}"-x" "cpp-output"
