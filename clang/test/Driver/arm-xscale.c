// RUN: %clang -target arm-freebsd -mcpu=xscale -### -c %s 2>&1 | FileCheck %s
// CHECK-NOT: error: the clang compiler does not support '-mcpu=xscale'
// CHECK: "-cc1"{{.*}} "-target-cpu" "xscale"{{.*}}
