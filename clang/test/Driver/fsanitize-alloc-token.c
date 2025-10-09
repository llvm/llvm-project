// RUN: %clang --target=x86_64-linux-gnu -fsanitize=alloc-token %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TOKEN-ALLOC
// CHECK-TOKEN-ALLOC: "-fsanitize=alloc-token"

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=alloc-token -fno-sanitize=alloc-token %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-TOKEN-ALLOC
// CHECK-NO-TOKEN-ALLOC-NOT: "-fsanitize=alloc-token"

// RUN: %clang --target=x86_64-linux-gnu -flto -fvisibility=hidden -fno-sanitize-ignorelist -fsanitize=alloc-token,undefined,cfi %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-COMPATIBLE
// CHECK-COMPATIBLE: "-fsanitize={{.*}}alloc-token"

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=alloc-token -fsanitize-minimal-runtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-MINIMAL
// CHECK-MINIMAL: "-fsanitize=alloc-token"
// CHECK-MINIMAL: "-fsanitize-minimal-runtime"

// RUN: %clang --target=arm-arm-non-eabi -fsanitize=alloc-token %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-BAREMETAL
// RUN: %clang --target=aarch64-none-elf -fsanitize=alloc-token %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-BAREMETAL
// CHECK-BAREMETAL: "-fsanitize=alloc-token"

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=alloc-token,address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-INCOMPATIBLE-ADDRESS
// CHECK-INCOMPATIBLE-ADDRESS: error: invalid argument '-fsanitize=alloc-token' not allowed with '-fsanitize=address'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=alloc-token,memory %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-INCOMPATIBLE-MEMORY
// CHECK-INCOMPATIBLE-MEMORY: error: invalid argument '-fsanitize=alloc-token' not allowed with '-fsanitize=memory'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=alloc-token -fsanitize-trap=alloc-token %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-INCOMPATIBLE-TRAP
// CHECK-INCOMPATIBLE-TRAP: error: unsupported argument 'alloc-token' to option '-fsanitize-trap='

// RUN: not %clang --target=x86_64-linux-gnu %s -fsanitize=alloc-token -fsanitize-recover=alloc-token -### 2>&1 | FileCheck %s --check-prefix=CHECK-INCOMPATIBLE-RECOVER
// CHECK-INCOMPATIBLE-RECOVER: unsupported argument 'alloc-token' to option '-fsanitize-recover='

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=alloc-token -fsanitize-alloc-token-fast-abi %s -### 2>&1 | FileCheck -check-prefix=CHECK-FASTABI %s
// CHECK-FASTABI: "-fsanitize-alloc-token-fast-abi"
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=alloc-token -fsanitize-alloc-token-fast-abi -fno-sanitize-alloc-token-fast-abi %s -### 2>&1 | FileCheck -check-prefix=CHECK-NOFASTABI %s
// CHECK-NOFASTABI-NOT: "-fsanitize-alloc-token-fast-abi"

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=alloc-token -fsanitize-alloc-token-extended %s -### 2>&1 | FileCheck -check-prefix=CHECK-EXTENDED %s
// CHECK-EXTENDED: "-fsanitize-alloc-token-extended"
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=alloc-token -fsanitize-alloc-token-extended -fno-sanitize-alloc-token-extended %s -### 2>&1 | FileCheck -check-prefix=CHECK-NOEXTENDED %s
// CHECK-NOEXTENDED-NOT: "-fsanitize-alloc-token-extended"

// RUN: %clang --target=x86_64-linux-gnu -falloc-token-max=0 -falloc-token-max=42 %s -### 2>&1 | FileCheck -check-prefix=CHECK-MAX %s
// CHECK-MAX: "-falloc-token-max=42"
// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=alloc-token -falloc-token-max=-1 %s 2>&1 | FileCheck -check-prefix=CHECK-INVALID-MAX %s
// CHECK-INVALID-MAX: error: invalid value
