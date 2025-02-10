// RUN: %clang -### -nogpuinc -nogpulib --offload-arch=gfx1030 --offload-arch=gfx1100 --offload-arch=gfx1101 -MD -MF tmp.d %s 2>&1 | FileCheck %s

// CHECK: Build config:
// CHECK-NOT: {{.*}}clang{{.*}}"-target-cpu" "gfx1030"{{.*}}"-dependency-file" "tmp.d"
// CHECK: {{.*}}lld{{.*}}"-plugin-opt=mcpu=gfx1030"
// CHECK-NOT: {{.*}}clang{{.*}}"-target-cpu" "gfx1100"{{.*}}"-dependency-file" "tmp.d"
// CHECK: {{.*}}lld{{.*}}"-plugin-opt=mcpu=gfx1100"
// CHECK-NOT: {{.*}}clang{{.*}}"-target-cpu" "gfx1101"{{.*}}"-dependency-file" "tmp.d"
// CHECK: {{.*}}lld{{.*}}"-plugin-opt=mcpu=gfx1101"
// CHECK: {{.*}}clang-offload-bundler
// CHECK: {{.*}}clang{{.*}}"-target-cpu" "x86-64"{{.*}}"-dependency-file" "tmp.d"

void main(){}
