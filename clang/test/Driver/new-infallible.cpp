// RUN: %clang -### -S -fno-new-infallible -fnew-infallible %s 2>&1 | FileCheck --check-prefix=NEW-INFALLIBLE %s
// NEW-INFALLIBLE: "-fnew-infallible"

// RUN: %clang -### -S -fnew-infallible -fno-new-infallible %s 2>&1 | FileCheck --check-prefix=NO-NEW-INFALLIBLE %s
// NO-NEW-INFALLIBLE-NOT: "-fnew-infallible"