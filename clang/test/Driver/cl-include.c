// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// RUN: %clang_cl -### -- %s 2>&1 | FileCheck %s --check-prefix=BUILTIN
// BUILTIN: "-internal-isystem" "{{.*lib.*clang.*include}}"

// RUN: %clang_cl -nobuiltininc -### -- %s 2>&1 | FileCheck %s --check-prefix=NOBUILTIN
// NOBUILTIN-NOT: "-internal-isystem" "{{.*lib.*clang.*include}}"

// RUN: env INCLUDE=/my/system/inc %clang_cl -### -- %s 2>&1 | FileCheck %s --check-prefix=STDINC
// STDINC: "-iexternal-after" "/my/system/inc"

// -nostdinc suppresses all of %INCLUDE%, clang resource dirs, and -imsvc dirs.
// RUN: env INCLUDE=/my/system/inc %clang_cl -nostdinc -imsvc /my/other/inc -### -- %s 2>&1 | FileCheck %s --check-prefix=NOSTDINC
// NOSTDINC: argument unused{{.*}}-imsvc
// NOSTDINC-NOT: "-iexternal-after" "/my/system/inc"
// NOSTDINC-NOT: "-internal-isystem" "{{.*lib.*clang.*include}}"

// /X suppresses %INCLUDE% but not clang resource dirs, -imsvc dirs, or /external: flags.
// RUN: env INCLUDE=/my/system/inc env FOO=/my/other/inc2 %clang_cl /X -imsvc /my/other/inc /external:env:FOO -### -- %s 2>&1 | FileCheck %s --check-prefix=SLASHX
// SLASHX-NOT: "argument unused{{.*}}-imsvc"
// SLASHX-NOT: "-external-after" "/my/system/inc"
// SLASHX: "-iexternal-env=FOO"
// SLASHX: "-internal-isystem" "{{.*lib.*clang.*include}}"
// SLASHX: "-internal-isystem" "/my/other/inc"

// /winsysroot does not suppress -imsvc dirs or /external: flags.
// RUN: env FOO=/my/other/inc2 %clang_cl /winsysroot /foo -imsvc /my/other/inc /external:env:FOO -### -- %s 2>&1 | FileCheck %s --check-prefix=SYSROOT
// SYSROOT-NOT: "argument unused{{.*}}-imsvc"
// SYSROOT-NOT: "argument unused{{.*}}/external:"
// SYSROOT: "-iexternal-env=FOO"
// SYSROOT: "-internal-isystem" "/my/other/inc"
// SYSROOT: "-internal-isystem" "/foo{{.*}}"

// RUN: env "FOO=/dir1;/dir2" env "BAR=/dir3" %clang_cl /external:env:FOO /external:env:BAR -### -- %s 2>&1 | FileCheck %s --check-prefix=EXTERNAL_ENV
// EXTERNAL_ENV: "-iexternal-env=FOO"
// EXTERNAL_ENV: "-iexternal-env=BAR"
