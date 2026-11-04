// REQUIRES: target={{s390x-ibm-zos}}

// In this case we expect __TARGET_LIB__ not to be defined because we don't
// include any files here, and in particular, any from the LE.
// RUN: %clang -mzos-target=current -dM -E %s | FileCheck --check-prefix=CURRENT %s
// CURRENT-NOT: #define __TARGET_LIB__

// RUN: %clang -mzos-target=zosv3r1 -dM -E %s | FileCheck --check-prefix=ZOSVR %s
// ZOSVR: #define __TARGET_LIB__ 0x43010000

// RUN: %clang -mzos-target=0x4301001f  -dM -E %s | FileCheck --check-prefix=HEX %s
// HEX: #define __TARGET_LIB__ 0x4301001F

// RUN: not %clang -mzos-target=0x42010000 -dM -E %s 2>&1 | FileCheck --check-prefix=ERR-DISCONTINUED %s
// ERR-DISCONTINUED: z/OS target level "0x42010000" is discontinued

// RUN: not %clang -mzos-target=Rubbish -dM -E %s 2>&1 | FileCheck --check-prefix=ERR-INVALID-ARG   %s
// ERR-INVALID-ARG: z/OS target level "Rubbish" is invalid
