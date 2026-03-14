// RUN: %clang -### --target=armv7-unknown-none-eabi -mcpu=cortex-m4 --sysroot= -fuse-ld=ld %s 2>&1 | FileCheck --check-prefix=NOLTO %s
// NOLTO: {{".*ld.*"}} {{.*}}
// NOLTO-NOT: "-plugin-opt=mcpu"

// RUN: %clang -### --target=armv7-unknown-none-eabi -mcpu=cortex-m4 --sysroot= -fuse-ld=ld -flto -O3 %s 2>&1 | FileCheck --check-prefix=LTO %s
// LTO: {{".*ld.*"}} {{.*}} "-plugin-opt=mcpu=cortex-m4" "-plugin-opt=O3"
