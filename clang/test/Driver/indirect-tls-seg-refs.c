// RUN: %clang -### -target x86_64-unknown-linux %s 2>&1 | FileCheck %s -check-prefix=TLSDIRECT
// RUN: %clang -### -target x86_64-unknown-linux -mno-tls-direct-seg-refs -mtls-direct-seg-refs %s 2>&1 | FileCheck %s -check-prefix=TLSDIRECT
// RUN: %clang -### -target x86_64-unknown-linux -mtls-direct-seg-refs -mno-tls-direct-seg-refs %s 2>&1 | FileCheck %s -check-prefix=NO-TLSDIRECT

// NO-TLSDIRECT: -mno-tls-direct-seg-refs
// TLSDIRECT-NOT: -mno-tls-direct-seg-refs
