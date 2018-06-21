// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// RUN: %clang_cl -### -- %s 2>&1 | FileCheck %s --check-prefix=BUILTIN
// BUILTIN: "-internal-isystem" "{{.*lib.*clang.*include}}"

// RUN: %clang_cl -nobuiltininc -### -- %s 2>&1 | FileCheck %s --check-prefix=NOBUILTIN
// NOBUILTIN-NOT: "-internal-isystem" "{{.*lib.*clang.*include}}"

// RUN: env INCLUDE=/my/system/inc %clang_cl -### -- %s 2>&1 | FileCheck %s --check-prefix=STDINC
// STDINC: "-internal-isystem" "/my/system/inc"

// RUN: env INCLUDE=/my/system/inc %clang_cl -nostdinc -### -- %s 2>&1 | FileCheck %s --check-prefix=NOSTDINC
// NOSTDINC-NOT: "-internal-isystem" "/my/system/inc"
