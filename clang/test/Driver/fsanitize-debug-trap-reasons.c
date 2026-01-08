// =============================================================================
// No Trap Reasons
// =============================================================================

// RUN: %clang -fsanitize=undefined -fsanitize-trap=undefined \
// RUN: -fsanitize-debug-trap-reasons=none %s -### 2>&1 | \
// RUN: FileCheck --check-prefix=NONE %s

// RUN: %clang -fsanitize=undefined -fsanitize-trap=undefined \
// RUN: -fno-sanitize-debug-trap-reasons %s -### 2>&1 | \
// RUN: FileCheck --check-prefix=NONE %s

// NONE: -fsanitize-debug-trap-reasons=none

// =============================================================================
// Basic Trap Reasons
// =============================================================================

// RUN: %clang -fsanitize=undefined -fsanitize-trap=undefined \
// RUN: -fsanitize-debug-trap-reasons=basic %s -### 2>&1 | \
// RUN: FileCheck --check-prefix=BASIC %s
// BASIC: -fsanitize-debug-trap-reasons=basic

// =============================================================================
// Detailed Trap Reasons
// =============================================================================

// RUN: %clang -fsanitize=undefined -fsanitize-trap=undefined \
// RUN: -fsanitize-debug-trap-reasons=detailed %s -### 2>&1 | \
// RUN: FileCheck --check-prefix=DETAILED %s

// RUN: %clang -fsanitize=undefined -fsanitize-trap=undefined \
// RUN: -fsanitize-debug-trap-reasons %s -### 2>&1 | \
// RUN: FileCheck --check-prefix=DETAILED %s

// DETAILED: -fsanitize-debug-trap-reasons=detailed

// =============================================================================
// Other cases
// =============================================================================

// By default the driver doesn't pass along any value and the default value is
// whatever is the default in CodeGenOptions.
// RUN: %clang %s -### 2>&1 | FileCheck --check-prefix=DEFAULT %s
// DEFAULT-NOT: -fsanitize-debug-trap-reasons

// Warning when not using UBSan
// RUN: %clang -fsanitize-debug-trap-reasons=none %s -### 2>&1 | \
// RUN: FileCheck --check-prefix=WARN %s
// WARN: warning: argument unused during compilation: '-fsanitize-debug-trap-reasons=none'

// Bad flag arguments are just passed along to the Frontend which handles rejecting
// invalid values. See `clang/test/Frontend/fsanitize-debug-trap-reasons.c`
// RUN: %clang -fsanitize=undefined -fsanitize-trap=undefined \
// RUN: -fsanitize-debug-trap-reasons=bad_value %s -### 2>&1 | \
// RUN: FileCheck --check-prefix=BAD_VALUE %s
// BAD_VALUE: -fsanitize-debug-trap-reasons=bad_value
