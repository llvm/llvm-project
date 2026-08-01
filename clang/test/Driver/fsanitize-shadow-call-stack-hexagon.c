// Test that -fsanitize=shadow-call-stack on Hexagon requires -ffixed-r19.

// RUN: not %clang --target=hexagon-unknown-linux-musl \
// RUN:   -fsanitize=shadow-call-stack %s -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=HEXAGON-SCS-NO-R19

// RUN: %clang --target=hexagon-unknown-linux-musl \
// RUN:   -fsanitize=shadow-call-stack -ffixed-r19 %s -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=HEXAGON-SCS-WITH-R19

// HEXAGON-SCS-NO-R19: error: invalid argument '-fsanitize=shadow-call-stack' only allowed with '-ffixed-r19'
// HEXAGON-SCS-WITH-R19: "-fsanitize=shadow-call-stack"
