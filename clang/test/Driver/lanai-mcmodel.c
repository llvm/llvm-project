// RUN: %clang --target=lanai -### -c -mcmodel=small %s 2>&1 | FileCheck --check-prefix=SMALL %s
// RUN: %clang --target=lanai -### -c -mcmodel=medium %s 2>&1 | FileCheck --check-prefix=MEDIUM %s
// RUN: %clang --target=lanai -### -c -mcmodel=large %s 2>&1 | FileCheck --check-prefix=LARGE %s
// RUN: not %clang --target=lanai -### -c -mcmodel=something %s 2>&1 | FileCheck --check-prefix=ERR-MCMODEL %s

// SMALL: "-mcmodel=small"
// MEDIUM: "-mcmodel=medium"
// LARGE: "-mcmodel=large"

// ERR-MCMODEL:  error: unsupported argument 'something' to option '-mcmodel=' for target 'lanai'
