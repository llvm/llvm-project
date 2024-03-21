/// When -fwrapv (implied by -fno-strict-overflow) is enabled,
/// -fsanitize=undefined does not expand to signed-integer-overflow.
/// -fsanitize=signed-integer-overflow is unaffected by -fwrapv.

// RUN: %clang -### --target=x86_64-linux -fwrapv -fsanitize=signed-integer-overflow %s 2>&1 | FileCheck %s
// CHECK: -fsanitize=signed-integer-overflow
// CHECK: -fsanitize-recover=signed-integer-overflow

// RUN: %clang -### --target=x86_64-linux -fno-strict-overflow -fsanitize=undefined %s 2>&1 | FileCheck %s --check-prefix=EXCLUDE
// RUN: %clang -### --target=x86_64-linux -fstrict-overflow -fwrapv -fsanitize=undefined %s 2>&1 | FileCheck %s --check-prefix=EXCLUDE
// EXCLUDE:     -fsanitize=alignment,array-bounds,
// EXCLUDE-NOT: signed-integer-overflow,
// EXCLUDE:      -fsanitize-recover=alignment,array-bounds,
// EXCLUDE-SAME: signed-integer-overflow

// RUN: %clang -### --target=x86_64-linux -fwrapv -fsanitize=undefined -fsanitize=signed-integer-overflow %s 2>&1 | FileCheck %s --check-prefix=INCLUDE
// RUN: %clang -### --target=x86_64-linux -fno-strict-overflow -fno-sanitize=signed-integer-overflow -fsanitize=undefined -fsanitize=signed-integer-overflow %s 2>&1 | FileCheck %s --check-prefix=INCLUDE
// INCLUDE:      -fsanitize=alignment,array-bounds,
// INCLUDE-SAME: signed-integer-overflow
// INCLUDE:      -fsanitize-recover=alignment,array-bounds,
// INCLUDE-SAME: signed-integer-overflow

/// -fsanitize-trap=undefined expands to signed-integer-overflow regardless of -fwrapv.
// RUN: %clang -### --target=x86_64-linux -fwrapv -fsanitize=undefined -fsanitize=signed-integer-overflow -fsanitize-trap=undefined %s 2>&1 | FileCheck %s --check-prefix=INCLUDE-TRAP
// INCLUDE-TRAP:      -fsanitize=alignment,array-bounds,
// INCLUDE-TRAP-SAME: signed-integer-overflow
// INCLUDE-TRAP:      -fsanitize-trap=alignment,array-bounds,
// INCLUDE-TRAP-SAME: signed-integer-overflow
