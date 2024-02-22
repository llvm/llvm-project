// RUN: %clang -target x86_64-apple-macos11 -fdepscan=daemon -c \
// RUN:   -x c %s -ccc-print-phases 2>&1 |                      \
// RUN:   FileCheck %s --check-prefix=CHECK-PHASES

// CHECK-PHASES: 0: input
// CHECK-PHASES: 1: depscan, {0}, response-file
// CHECK-PHASES: 2: preprocessor, {1}, cpp-output
// CHECK-PHASES: 3: compiler, {2}, ir
// CHECK-PHASES: 4: backend, {3}, assembler
// CHECK-PHASES: 5: assembler, {4}, object
// CHECK-PHASES: 6: bind-arch, "x86_64", {5}, object

// RUN: %clang -target x86_64-apple-macos11 -fdepscan=daemon -c \
// RUN:   -x c %s -ccc-print-bindings 2>&1 |                    \
// RUN:   FileCheck %s --check-prefix=CHECK-BINDINGS

// CHECK-BINDINGS: # "x86_64-apple-macos11" - "clang"
// CHECK-BINDINGS: # "x86_64-apple-macos11" - "clang"


// RUN: %clang -target x86_64-apple-macos11 -fdepscan=daemon -c \
// RUN:   -x c %s -### 2>&1 |                                   \
// RUN:   FileCheck %s --check-prefix=CHECK-CMD

// CHECK-CMD: "-cc1depscan" "-fdepscan=daemon" "-o"
// CHECK-CMD: "-cc1" "@

// RUN: %clang -target x86_64-apple-macos11 -fdepscan=daemon -c \
// RUN:   -x c %s -save-temps -###

