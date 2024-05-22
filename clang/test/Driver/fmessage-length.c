// RUN: %clang -### -c %s -fmessage-length=80 2>&1 | FileCheck %s
// CHECK: "-fmessage-length=80"

/// Omit -fmessage-length=0 to simplify common CC1 command lines.
// RUN: %clang -### -c %s -fmessage-length=0 2>&1 | FileCheck --check-prefix=ZERO %s
// ZERO-NOT: "-fmessage-length=0"

// RUN: not %clang -### -c %s -fmessage-length=nan 2>&1 | FileCheck --check-prefix=ERR %s
// ERR: error: invalid argument 'nan' to -fmessage-length=
