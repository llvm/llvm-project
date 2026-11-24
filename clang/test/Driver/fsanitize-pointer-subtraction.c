/// When -fdefined-pointer-subtraction is enabled, -fsanitize=undefined does not
/// expand to pointer-subtraction.
/// -fsanitize=pointer-subtraction is unaffected by
/// -fdefined-pointer-subtraction.

// RUN: %clang -### --target=x86_64-linux -fdefined-pointer-subtraction -fsanitize=pointer-subtraction %s 2>&1 | FileCheck %s
// CHECK: "-fsanitize=pointer-subtraction"
// CHECK: "-fsanitize-recover=pointer-subtraction"

// RUN: %clang -### --target=x86_64-linux -fdefined-pointer-subtraction -fsanitize=undefined %s 2>&1 | FileCheck %s --check-prefix=EXCLUDE
// EXCLUDE:     "-fsanitize=alignment,array-bounds,
// EXCLUDE-NOT: {{[=,]}}pointer-subtraction
// EXCLUDE:     "-fsanitize-merge=alignment,array-bounds,

// RUN: %clang -### --target=x86_64-linux -fdefined-pointer-subtraction -fsanitize=undefined -fsanitize=pointer-subtraction %s 2>&1 | FileCheck %s --check-prefix=INCLUDE
// INCLUDE: "-fsanitize={{[^"]*}}pointer-subtraction
// INCLUDE: "-fsanitize-recover={{[^"]*}}pointer-subtraction
