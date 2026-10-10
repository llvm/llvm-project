/// When -fdefined-pointer-subtraction is enabled, -fsanitize=undefined does not
/// expand to unaligned-pointer-subtraction.
/// -fsanitize=unaligned-pointer-subtraction is unaffected by
/// -fdefined-pointer-subtraction.

// RUN: %clang -### --target=x86_64-linux -fdefined-pointer-subtraction -fsanitize=unaligned-pointer-subtraction %s 2>&1 | FileCheck %s
// CHECK: "-fsanitize=unaligned-pointer-subtraction"
// CHECK: "-fsanitize-recover=unaligned-pointer-subtraction"

// RUN: %clang -### --target=x86_64-linux -fdefined-pointer-subtraction -fsanitize=undefined %s 2>&1 | FileCheck %s --check-prefix=EXCLUDE
// EXCLUDE:     "-fsanitize=alignment,array-bounds,
// EXCLUDE-NOT: unaligned-pointer-subtraction
// EXCLUDE:     "-fsanitize-merge=alignment,array-bounds,

// RUN: %clang -### --target=x86_64-linux -fdefined-pointer-subtraction -fsanitize=undefined -fsanitize=unaligned-pointer-subtraction %s 2>&1 | FileCheck %s --check-prefix=INCLUDE
// INCLUDE: "-fsanitize={{[^"]*}}unaligned-pointer-subtraction
// INCLUDE: "-fsanitize-recover={{[^"]*}}unaligned-pointer-subtraction
