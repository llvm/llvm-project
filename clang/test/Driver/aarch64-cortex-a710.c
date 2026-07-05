// RUN: %clang --target=aarch64 -mcpu=cortex-a710 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A710 %s
// CORTEX-A710: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-a710"
// CORTEX-A710-NOT: "-target-feature" "{{[+-]}}sm4"
// CORTEX-A710-NOT: "-target-feature" "{{[+-]}}sha3"
// CORTEX-A710-NOT: "-target-feature" "{{[+-]}}aes"
// CORTEX-A710-SAME: {{$}}
// RUN: %clang --target=aarch64 -mcpu=cortex-a710+crypto -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A710-CRYPTO %s
// CORTEX-A710-CRYPTO: "-cc1"{{.*}} "-triple" "aarch64"{{.*}} "-target-feature" "+aes"{{.*}} "-target-feature" "+sha2" "-target-feature" "+sha3"{{.*}} "-target-feature" "+sm4"
