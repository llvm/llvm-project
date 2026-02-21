// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// RUN: %clang_cl -### -- %s 2>&1 | FileCheck %s --check-prefix=BUILTIN
// BUILTIN: "-internal-isystem" "{{.*lib.*clang.*include}}"

// RUN: %clang_cl -nobuiltininc -### -- %s 2>&1 | FileCheck %s --check-prefix=NOBUILTIN
// NOBUILTIN-NOT: "-internal-isystem" "{{.*lib.*clang.*include}}"

// RUN: env INCLUDE=/my/system/inc env EXTERNAL_INCLUDE=/my/system/inc2 %clang_cl -### -- %s 2>&1 | FileCheck %s --check-prefix=STDINC
// STDINC: "-internal-isystem" "{{.*lib.*clang.*include}}"
// STDINC: "-internal-isystem" "/my/system/inc"
// STDINC: "-internal-iexternal-system" "/my/system/inc2"

// -nostdinc suppresses all of %INCLUDE%, clang resource dirs, and -imsvc dirs.
// RUN: env INCLUDE=/my/system/inc1 env EXTERNAL_INCLUDE=/my/system/inc2 %clang_cl -nostdinc -imsvc /my/other/inc -### -- %s 2>&1 | FileCheck %s --check-prefix=NOSTDINC
// NOSTDINC: argument unused{{.*}}-imsvc
// NOSTDINC-NOT: "-internal-isystem" "{{.*lib.*clang.*include}}"
// NOSTDINC-NOT: "/my/other/inc"
// NOSTDINC-NOT: "/my/system/inc1"
// NOSTDINC-NOT: "/my/system/inc2"

// /X suppresses %INCLUDE% and %EXTERNAL_INCLUDE% but not clang resource dirs, -imsvc dirs, or /external: flags.
// RUN: env INCLUDE=/my/system/inc1 env EXTERNAL_INCLUDE=/my/system/inc2 env FOO=/my/other/inc2 %clang_cl /X -imsvc /my/other/inc /external:env:FOO -### -- %s 2>&1 | FileCheck %s --check-prefix=SLASHX
// SLASHX-NOT: "argument unused{{.*}}-imsvc"
// SLASHX-NOT: "/my/system/inc1"
// SLASHX-NOT: "/my/system/inc2"
// SLASHX: "-iexternal-system" "/my/other/inc2"
// SLASHX: "-internal-isystem" "{{.*lib.*clang.*include}}"
// SLASHX: "-internal-isystem" "/my/other/inc"

// /winsysroot suppresses %INCLUDE% and %EXTERNAL_INCLUDE% but not -imsvc dirs or /external: flags.
// RUN: env INCLUDE=/my/system/inc1 env EXTERNAL_INCLUDE=/my/system/inc2 env FOO=/my/other/inc2 %clang_cl /winsysroot /foo -imsvc /my/other/inc /external:env:FOO -### -- %s 2>&1 | FileCheck %s --check-prefix=SYSROOT
// SYSROOT-NOT: "argument unused{{.*}}-imsvc"
// SYSROOT-NOT: "argument unused{{.*}}/external:"
// SYSROOT-NOT: "/my/system/inc1"
// SYSROOT-NOT: "/my/system/inc2"
// SYSROOT: "-iexternal-system" "/my/other/inc2"
// SYSROOT: "-internal-isystem" "/my/other/inc"
// SYSROOT: "-internal-isystem" "/foo{{.*}}"

// RUN: env "FOO=/dir1%{pathsep}/dir2" env "BAR=/dir3" %clang_cl /external:env:FOO /external:env:BAR -### -- %s 2>&1 | FileCheck %s --check-prefix=EXTERNAL_ENV
// EXTERNAL_ENV: "-iexternal-system" "/dir1"
// EXTERNAL_ENV: "-iexternal-system" "/dir2"
// EXTERNAL_ENV: "-iexternal-system" "/dir3"

// /winsdkversion suppresses %INCLUDE% and %EXTERNAL_INCLUDE%
// RUN:env INCLUDE=/my/system/inc env EXTERNAL_INCLUDE=/my/system/inc2 %clang_cl /winsdkversion 99.99.9999.9 -### -- %s 2>&1 | FileCheck %s --check-prefix=SDKVERSION
// SDKVERSION-NOT: "-internal-isystem" "/my/system/inc"
// SDKVERSION-NOT: "-iexternal-system" "/my/system/inc2"
