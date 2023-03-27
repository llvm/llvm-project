// RUN: %clang --target=aarch64 -mcpu=cortex-a510 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A510 %s
// CORTEX-A510: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-a510"
// CORTEX-A510-NOT: "-target-feature" "{{[+-]}}sm4"
// CORTEX-A510-NOT: "-target-feature" "{{[+-]}}sha3"
// CORTEX-A510-NOT: "-target-feature" "{{[+-]}}aes"
// CORTEX-A510-SAME: {{$}}
// RUN: %clang --target=aarch64 -mcpu=cortex-a510+crypto -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A510-CRYPTO %s
// CORTEX-A510-CRYPTO: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+sm4" "-target-feature" "+sha3" "-target-feature" "+sha2" "-target-feature" "+aes"
