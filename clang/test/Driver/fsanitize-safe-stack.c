// RUN: %clang -fno-sanitize=safe-stack -### %s 2>&1 | FileCheck %s -check-prefix=NOSP
// NOSP-NOT: "-fsanitize=safe-stack"

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=safe-stack -### %s 2>&1 | FileCheck %s -check-prefix=NO-SP
// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=address,safe-stack -### %s 2>&1 | FileCheck %s -check-prefix=SP-ASAN
// RUN: %clang --target=x86_64-linux-gnu -fstack-protector -fsanitize=safe-stack -### %s 2>&1 | FileCheck %s -check-prefix=SP
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=safe-stack -fstack-protector-all -### %s 2>&1 | FileCheck %s -check-prefix=SP
// RUN: %clang --target=arm-linux-androideabi -fsanitize=safe-stack -### %s 2>&1 | FileCheck %s -check-prefix=NO-SP
// RUN: %clang --target=aarch64-linux-android -fsanitize=safe-stack -### %s 2>&1 | FileCheck %s -check-prefix=NO-SP
// NO-SP-NOT: stack-protector
// NO-SP: "-fsanitize=safe-stack"
// SP-ASAN: error: invalid argument '-fsanitize=safe-stack' not allowed with '-fsanitize=address'
// SP: -stack-protector
// SP: "-fsanitize=safe-stack"
// NO-SP-NOT: stack-protector
