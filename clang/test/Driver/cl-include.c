// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// RUN: %clang_cl -### -- %s 2>&1 | FileCheck %s --check-prefix=BUILTIN
// BUILTIN: "-internal-isystem" "{{.*lib.*clang.*include}}"

// RUN: %clang_cl -nobuiltininc -### -- %s 2>&1 | FileCheck %s --check-prefix=NOBUILTIN
// NOBUILTIN-NOT: "-internal-isystem" "{{.*lib.*clang.*include}}"

// RUN: env INCLUDE=/my/system/inc env EXTERNAL_INCLUDE=/my/system/inc2 %clang_cl -### -- %s 2>&1 | FileCheck %s --check-prefix=STDINC
// STDINC: "-internal-isystem" "/my/system/inc"
// STDINC: "-internal-isystem" "/my/system/inc2"

// -nostdinc suppresses all of %INCLUDE%, clang resource dirs, and -imsvc dirs.
// RUN: env INCLUDE=/my/system/inc env EXTERNAL_INCLUDE=/my/system/inc2 %clang_cl -nostdinc -imsvc /my/other/inc -### -- %s 2>&1 | FileCheck %s --check-prefix=NOSTDINC
// NOSTDINC: argument unused{{.*}}-imsvc
// NOSTDINC-NOT: "-internal-isystem" "/my/system/inc"
// NOSTDINC-NOT: "-internal-isystem" "/my/system/inc2"
// NOSTDINC-NOT: "-internal-isystem" "{{.*lib.*clang.*include}}"
// NOSTDINC-NOT: "-internal-isystem" "/my/other/inc"

// /X suppresses %INCLUDE% and %EXTERNAL_INCLUDE% but not clang resource dirs, -imsvc dirs, or /external: flags.
// RUN: env INCLUDE=/my/system/inc env EXTERNAL_INCLUDE=/my/system/inc2 env FOO=/my/other/inc2 %clang_cl /X -imsvc /my/other/inc /external:env:FOO -### -- %s 2>&1 | FileCheck %s --check-prefix=SLASHX
// SLASHX-NOT: "argument unused{{.*}}-imsvc"
// SLASHX-NOT: "-internal-isystem" "/my/system/inc"
// SLASHX-NOT: "-internal-isystem" "/my/system/inc2"
// SLASHX: "-internal-isystem" "{{.*lib.*clang.*include}}"
// SLASHX: "-internal-isystem" "/my/other/inc"
// SLASHX: "-internal-isystem" "/my/other/inc2"

// /winsysroot suppresses %EXTERNAL_INCLUDE% but not -imsvc dirs or /external: flags.
// RUN: env env EXTERNAL_INCLUDE=/my/system/inc env FOO=/my/other/inc2 %clang_cl /winsysroot /foo -imsvc /my/other/inc /external:env:FOO -### -- %s 2>&1 | FileCheck %s --check-prefix=SYSROOT
// SYSROOT-NOT: "argument unused{{.*}}-imsvc"
// SYSROOT-NOT: "argument unused{{.*}}/external:"
// SYSROOT-NOT: "/my/system/inc"
// SYSROOT: "-internal-isystem" "/my/other/inc"
// SYSROOT: "-internal-isystem" "/my/other/inc2"
// SYSROOT: "-internal-isystem" "/foo{{.*}}"

// RUN: env "FOO=/dir1;/dir2" env "BAR=/dir3" %clang_cl /external:env:FOO /external:env:BAR -### -- %s 2>&1 | FileCheck %s --check-prefix=EXTERNAL_ENV
// EXTERNAL_ENV: "-internal-isystem" "/dir1"
// EXTERNAL_ENV: "-internal-isystem" "/dir2"
// EXTERNAL_ENV: "-internal-isystem" "/dir3"
