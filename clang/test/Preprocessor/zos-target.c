// REQUIRES: target={{s390x-ibm-zos}}

// In this case we expect __TARGET_LIB__ not to be defined because we don't
// include any files here, and in particular, any from the LE.
// RUN: %clang -mzos-target=current -dM -E %s | FileCheck --check-prefix=CURRENT %s
// CURRENT-NOT: #define __TARGET_LIB__

// RUN: %clang -mzos-target=zosv2r5 -dM -E %s | FileCheck --check-prefix=ZOSVR %s
// ZOSVR: #define __TARGET_LIB__ 0x42050000

// RUN: %clang -mzos-target=0x4204001f  -dM -E %s | FileCheck --check-prefix=HEX %s
// HEX: #define __TARGET_LIB__ 0x4204001F

// RUN: not %clang -mzos-target=0x42010000 -dM -E %s 2>&1 | FileCheck --check-prefix=ERR-DISCONTINUED %s
// ERR-DISCONTINUED: z/OS target level "0x42010000" is discontinued. Unexpected behavior might occur if an out-of-support target level is specified. Use z/OS target level "zosv2r4", or later instead

// RUN: not %clang -mzos-target=Rubbish -dM -E %s 2>&1 | FileCheck --check-prefix=ERR-INVALID-ARG   %s
// ERR-INVALID-ARG: "Rubbish" is not recognized as a valid z/OS target level. The z/OS target level must be "current", or of the form "zosvVrR", where "V" is the version and "R" is the release, or given as a "0x"-prefixed eight digit hexadecimal value
