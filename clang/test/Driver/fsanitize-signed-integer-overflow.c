/// -fsanitize=signed-integer-overflow is a group of
/// signed-integer-overflow-arith and signed-integer-overflow-divide.

// RUN: %clang -### --target=x86_64-linux -fsanitize=signed-integer-overflow %s 2>&1 | FileCheck %s
// RUN: %clang -### --target=x86_64-linux -fsanitize=signed-integer-overflow-arith,signed-integer-overflow-divide %s 2>&1 | FileCheck %s
// CHECK: "-fsanitize=signed-integer-overflow-arith,signed-integer-overflow-divide"
// CHECK: "-fsanitize-recover=signed-integer-overflow-arith,signed-integer-overflow-divide"

// RUN: %clang -### --target=x86_64-linux -fsanitize=signed-integer-overflow-arith %s 2>&1 | FileCheck %s --check-prefix=ARITH
// RUN: %clang -### --target=x86_64-linux -fsanitize=signed-integer-overflow -fno-sanitize=signed-integer-overflow-divide %s 2>&1 | FileCheck %s --check-prefix=ARITH
// ARITH: "-fsanitize=signed-integer-overflow-arith"
// ARITH: "-fsanitize-recover=signed-integer-overflow-arith"

// RUN: %clang -### --target=x86_64-linux -fsanitize=signed-integer-overflow-divide %s 2>&1 | FileCheck %s --check-prefix=DIVIDE
// RUN: %clang -### --target=x86_64-linux -fsanitize=signed-integer-overflow -fno-sanitize=signed-integer-overflow-arith %s 2>&1 | FileCheck %s --check-prefix=DIVIDE
// DIVIDE: "-fsanitize=signed-integer-overflow-divide"
// DIVIDE: "-fsanitize-recover=signed-integer-overflow-divide"

/// The recover, trap and merge lists accept the group and the leaves.
// RUN: %clang -### --target=x86_64-linux -fsanitize=signed-integer-overflow -fno-sanitize-recover=signed-integer-overflow -fsanitize-recover=signed-integer-overflow-divide -fsanitize-trap=signed-integer-overflow-arith -fno-sanitize-merge=signed-integer-overflow %s 2>&1 | FileCheck %s --check-prefix=MIXED
// MIXED:     "-fsanitize=signed-integer-overflow-arith,signed-integer-overflow-divide"
// MIXED:     "-fsanitize-recover=signed-integer-overflow-divide"
// MIXED:     "-fsanitize-trap=signed-integer-overflow-arith"
// MIXED-NOT: "-fsanitize-merge

/// When -fwrapv (implied by -fno-strict-overflow) is enabled,
/// -fsanitize=undefined does not expand to signed-integer-overflow.
/// -fsanitize=signed-integer-overflow is unaffected by -fwrapv.

// RUN: %clang -### --target=x86_64-linux -fwrapv -fsanitize=signed-integer-overflow %s 2>&1 | FileCheck %s

// RUN: %clang -### --target=x86_64-linux -fno-strict-overflow -fsanitize=undefined %s 2>&1 | FileCheck %s --check-prefix=EXCLUDE
// RUN: %clang -### --target=x86_64-linux -fstrict-overflow -fwrapv -fsanitize=undefined %s 2>&1 | FileCheck %s --check-prefix=EXCLUDE
// EXCLUDE:     -fsanitize=alignment,array-bounds,
// EXCLUDE-NOT: {{[=,]}}signed-integer-overflow
// EXCLUDE:     -fsanitize-recover=alignment,array-bounds,
// EXCLUDE-NOT: {{[=,]}}signed-integer-overflow

// RUN: %clang -### --target=x86_64-linux -fwrapv -fsanitize=undefined -fsanitize=signed-integer-overflow %s 2>&1 | FileCheck %s --check-prefix=INCLUDE
// RUN: %clang -### --target=x86_64-linux -fno-strict-overflow -fno-sanitize=signed-integer-overflow -fsanitize=undefined -fsanitize=signed-integer-overflow %s 2>&1 | FileCheck %s --check-prefix=INCLUDE
// INCLUDE:      -fsanitize=alignment,array-bounds,
// INCLUDE-SAME: signed-integer-overflow-arith,signed-integer-overflow-divide,
// INCLUDE:      -fsanitize-recover=alignment,array-bounds,
// INCLUDE-SAME: signed-integer-overflow-arith,signed-integer-overflow-divide,

/// -fsanitize-trap=undefined expands to signed-integer-overflow regardless of -fwrapv.
// RUN: %clang -### --target=x86_64-linux -fwrapv -fsanitize=undefined -fsanitize=signed-integer-overflow -fsanitize-trap=undefined %s 2>&1 | FileCheck %s --check-prefix=INCLUDE-TRAP
// INCLUDE-TRAP:      -fsanitize=alignment,array-bounds,
// INCLUDE-TRAP-SAME: signed-integer-overflow-arith,signed-integer-overflow-divide,
// INCLUDE-TRAP:      -fsanitize-trap=alignment,array-bounds,
// INCLUDE-TRAP-SAME: signed-integer-overflow-arith,signed-integer-overflow-divide,

/// Disabling both leaves clears the group so no empty -fsanitize= or sanitizer
/// preprocessor define remains.
// RUN: %clang -### --target=x86_64-unknown-netbsd -fsanitize=signed-integer-overflow -fno-sanitize=undefined %s 2>&1 | FileCheck %s --check-prefix=DISABLED
// RUN: %clang -### --target=x86_64-unknown-netbsd -fsanitize=signed-integer-overflow -fno-sanitize=integer %s 2>&1 | FileCheck %s --check-prefix=DISABLED
// RUN: %clang -### --target=x86_64-unknown-netbsd -fsanitize=signed-integer-overflow -fno-sanitize=signed-integer-overflow-arith,signed-integer-overflow-divide %s 2>&1 | FileCheck %s --check-prefix=DISABLED
// DISABLED:     "-cc1"
// DISABLED-NOT: "-fsanitize=
// DISABLED-NOT: "-D_REENTRANT"
