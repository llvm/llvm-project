// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// RUN: %clang_cl -### -- %s 2>&1 | FileCheck %s --check-prefix=BUILTIN
// BUILTIN: "-internal-isystem" "{{.*lib.*clang.*include}}"

// RUN: %clang_cl -nobuiltininc -### -- %s 2>&1 | FileCheck %s --check-prefix=NOBUILTIN
// NOBUILTIN-NOT: "-internal-isystem" "{{.*lib.*clang.*include}}"

// RUN: env INCLUDE=/my/system/inc %clang_cl -### -- %s 2>&1 | FileCheck %s --check-prefix=STDINC
// STDINC: "-internal-isystem" "/my/system/inc"

// -nostdinc suppresses all of %INCLUDE%, clang resource dirs, and -imsvc dirs.
// RUN: env INCLUDE=/my/system/inc %clang_cl -nostdinc -imsvc /my/other/inc -### -- %s 2>&1 | FileCheck %s --check-prefix=NOSTDINC
// NOSTDINC: argument unused{{.*}}-imsvc
// NOSTDINC-NOT: "-internal-isystem" "/my/system/inc"
// NOSTDINC-NOT: "-internal-isystem" "{{.*lib.*clang.*include}}"
// NOSTDINC-NOT: "-internal-isystem" "/my/other/inc"

// /X suppresses %INCLUDE% but not clang resource dirs or -imsvc dirs.
// RUN: env INCLUDE=/my/system/inc %clang_cl /X -imsvc /my/other/inc -### -- %s 2>&1 | FileCheck %s --check-prefix=SLASHX
// SLASHX-NOT: "argument unused{{.*}}-imsvc"
// SLASHX-NOT: "-internal-isystem" "/my/system/inc"
// SLASHX: "-internal-isystem" "{{.*lib.*clang.*include}}"
// SLASHX: "-internal-isystem" "/my/other/inc"
