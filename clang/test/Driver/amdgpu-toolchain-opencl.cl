// RUN: %clang -### -target amdgcn-amd-amdhsa-opencl -x cl -c -emit-llvm -mcpu=fiji -O0 %s 2>&1 | FileCheck -check-prefix=CHECK_O0 %s
// RUN: %clang -### -target amdgcn-amd-amdhsa-opencl -x cl -c -emit-llvm -mcpu=fiji -O1 %s 2>&1 | FileCheck -check-prefix=CHECK_O1 %s
// RUN: %clang -### -target amdgcn-amd-amdhsa-opencl -x cl -c -emit-llvm -mcpu=fiji -O2 %s 2>&1 | FileCheck -check-prefix=CHECK_O2 %s
// RUN: %clang -### -target amdgcn-amd-amdhsa-opencl -x cl -c -emit-llvm -mcpu=fiji -O3 %s 2>&1 | FileCheck -check-prefix=CHECK_O3 %s
// RUN: %clang -### -target amdgcn-amd-amdhsa-opencl -x cl -c -emit-llvm -mcpu=fiji -O4 %s 2>&1 | FileCheck -check-prefix=CHECK_O4 %s
// RUN: %clang -### -target amdgcn-amd-amdhsa-opencl -x cl -c -emit-llvm -mcpu=fiji -O5 %s 2>&1 | FileCheck -check-prefix=CHECK_O5 %s
// RUN: %clang -### -target amdgcn-amd-amdhsa-opencl -x cl -c -emit-llvm -mcpu=fiji -Og %s 2>&1 | FileCheck -check-prefix=CHECK_Og %s
// RUN: %clang -### -target amdgcn-amd-amdhsa-opencl -x cl -c -emit-llvm -mcpu=fiji -Ofast %s 2>&1 | FileCheck -check-prefix=CHECK_Ofast %s
// RUN: %clang -### -target amdgcn-amd-amdhsa-opencl -x cl -c -emit-llvm -mcpu=fiji %s 2>&1 | FileCheck -check-prefix=CHECK_O_DEFAULT %s
// CHECK_O0: clang{{.*}} "-O0"
// CHECK_O1: clang{{.*}} "-O1"
// CHECK_O2: clang{{.*}} "-O2"
// CHECK_O3: clang{{.*}} "-O3"
// CHECK_O4: clang{{.*}} "-O3"
// CHECK_O5: clang{{.*}} "-O5"
// CHECK_Og: clang{{.*}} "-Og"
// CHECK_Ofast: {{.*}}clang{{.*}} "-Ofast"
// CHECK_O_DEFAULT: clang{{.*}} "-O3"

