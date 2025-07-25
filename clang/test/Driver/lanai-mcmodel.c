// RUN: %clang --target=lanai -### -c -mcmodel=small %s 2>&1 | FileCheck --check-prefix=SMALL %s
// RUN: %clang --target=lanai -### -c -mcmodel=medium %s 2>&1 | FileCheck --check-prefix=MEDIUM %s
// RUN: %clang --target=lanai -### -c -mcmodel=large %s 2>&1 | FileCheck --check-prefix=LARGE %s

// SMALL: "-mcmodel=small"
// MEDIUM: "-mcmodel=medium"
// LARGE: "-mcmodel=large"
