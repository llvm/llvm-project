// RUN: not %clang --target=aarch64-linux-android -fsanitize=memtag,address -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANMT-SANA
// CHECK-SANMT-SANA: '-fsanitize=memtag' not allowed with '-fsanitize=address'

// RUN: not %clang --target=aarch64-linux-android -fsanitize=memtag,hwaddress -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANMT-SANHA
// CHECK-SANMT-SANHA: '-fsanitize=memtag' not allowed with '-fsanitize=hwaddress'

// RUN: not %clang --target=i386-linux-android -fsanitize=memtag -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANMT-BAD-ARCH
// RUN: not %clang --target=x86_64-linux-android -fsanitize=memtag -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANMT-BAD-ARCH
// CHECK-SANMT-BAD-ARCH: unsupported option '-fsanitize=memtag' for target

// RUN: %clang --target=aarch64-linux-android31 -fsanitize=memtag -march=armv8-a+memtag %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANMT-MT
// CHECK-SANMT-MT: "-target-feature" "+mte"
// CHECK-SANMT-MT-SAME: "-fsanitize=memtag-stack,memtag-heap,memtag-globals"

// RUN: not %clang --target=aarch64-linux -fsanitize=memtag -Xclang -target-feature -Xclang +mte %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANMT-MT

// RUN: not %clang --target=aarch64-linux -fsanitize=memtag %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANMT-NOMT-0
// CHECK-SANMT-NOMT-0: '-fsanitize=memtag-stack' requires hardware support (+memtag)

// RUN: not %clang --target=aarch64-linux -fsanitize=memtag -I +mte %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANMT-NOMT-1
// CHECK-SANMT-NOMT-1: '-fsanitize=memtag-stack' requires hardware support (+memtag)

// RUN: not %clang --target=aarch64-linux-android31 -fsanitize-trap=memtag -march=armv8-a+memtag -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANMT-TRAP
// CHECK-SANMT-TRAP: error: unsupported argument 'memtag' to option '-fsanitize-trap='
