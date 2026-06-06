// RUN: %clang --target=aarch64 -mcpu=cortex-x2 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-X2 %s
// CORTEX-X2: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-x2"
// CORTEX-X2-NOT: "-target-feature" "{{[+-]}}sm4"
// CORTEX-X2-NOT: "-target-feature" "{{[+-]}}sha3"
// CORTEX-X2-NOT: "-target-feature" "{{[+-]}}aes"
// CORTEX-X2-SAME: {{$}}
// RUN: %clang --target=aarch64 -mcpu=cortex-x2+crypto -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-X2-CRYPTO %s
// CORTEX-X2-CRYPTO: "-cc1"{{.*}} "-triple" "aarch64"{{.*}} "-target-feature" "+aes"{{.*}} "-target-feature" "+sha2" "-target-feature" "+sha3"{{.*}} "-target-feature" "+sm4"
