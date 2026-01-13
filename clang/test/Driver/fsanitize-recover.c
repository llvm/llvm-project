// RUN: %clang --target=x86_64-linux-gnu %s -fsanitize=undefined -### 2>&1 | FileCheck %s --check-prefix=CHECK-RECOVER-UBSAN
// RUN: %clang --target=x86_64-linux-gnu %s -fsanitize=undefined -fsanitize-recover -### 2>&1 | FileCheck %s --check-prefix=CHECK-RECOVER-UBSAN
// RUN: %clang --target=x86_64-linux-gnu %s -fsanitize=undefined -fsanitize-recover=all -### 2>&1 | FileCheck %s --check-prefix=CHECK-RECOVER-UBSAN
// RUN: %clang --target=x86_64-linux-gnu %s -fsanitize=undefined -fno-sanitize-recover -fsanitize-recover=undefined -### 2>&1 | FileCheck %s --check-prefix=CHECK-RECOVER-UBSAN
// RUN: %clang --target=x86_64-linux-gnu %s -fsanitize=undefined -fno-sanitize-recover=undefined -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-RECOVER-UBSAN
// RUN: %clang --target=x86_64-linux-gnu %s -fsanitize=undefined -fno-sanitize-recover=all -fsanitize-recover=thread -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-RECOVER-UBSAN
// RUN: %clang --target=x86_64-linux-gnu %s -fsanitize=undefined -fsanitize-recover=all -fno-sanitize-recover=undefined -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-RECOVER-UBSAN
// CHECK-RECOVER-UBSAN: "-fsanitize-recover={{((signed-integer-overflow|integer-divide-by-zero|function|shift-base|shift-exponent|vla-bound|alignment|null|pointer-overflow|float-cast-overflow|array-bounds|enum|bool|builtin|returns-nonnull-attribute|nonnull-attribute),?){16}"}}
// CHECK-NO-RECOVER-UBSAN-NOT: -fsanitize-recover

// RUN: %clang --target=x86_64-linux-gnu %s -fsanitize=undefined -fno-sanitize-recover=all -fsanitize-recover=object-size,shift-base -### 2>&1 | FileCheck %s --check-prefix=CHECK-PARTIAL-RECOVER
// CHECK-PARTIAL-RECOVER: "-fsanitize-recover={{((shift-base),?){1}"}}

// RUN: %clang --target=x86_64-linux-gnu %s -fsanitize=address -fsanitize-recover=all -### 2>&1 | FileCheck %s --check-prefix=CHECK-RECOVER-ASAN
// RUN: %clang --target=x86_64-linux-gnu %s -fsanitize=address -fsanitize-recover -### 2>&1 | FileCheck %s --check-prefix=CHECK-RECOVER-ASAN
// CHECK-RECOVER-ASAN: "-fsanitize-recover=address"

// RUN: not %clang --target=x86_64-linux-gnu %s -fsanitize=undefined -fsanitize-recover=foobar,object-size,unreachable -### 2>&1 | FileCheck %s --check-prefix=CHECK-DIAG-RECOVER
// CHECK-DIAG-RECOVER: unsupported argument 'foobar' to option '-fsanitize-recover='
// CHECK-DIAG-RECOVER: unsupported argument 'unreachable' to option '-fsanitize-recover='

// RUN: %clang --target=x86_64-linux-gnu %s -fsanitize=undefined -fsanitize-recover -fno-sanitize-recover -### 2>&1 | FileCheck %s --check-prefix=CHECK-DEPRECATED-RECOVER
// CHECK-DEPRECATED-RECOVER-NOT: is deprecated

// RUN: not %clang --target=x86_64-linux-gnu %s -fsanitize=kernel-address -fno-sanitize-recover=kernel-address -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-RECOVER-KASAN
// RUN: not %clang --target=x86_64-linux-gnu %s -fsanitize=kernel-hwaddress -fno-sanitize-recover=kernel-hwaddress -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-RECOVER-KHWASAN
// CHECK-NO-RECOVER-KASAN: unsupported argument 'kernel-address' to option '-fno-sanitize-recover='
// CHECK-NO-RECOVER-KHWASAN: unsupported argument 'kernel-hwaddress' to option '-fno-sanitize-recover='
