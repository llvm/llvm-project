// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// First check that regular clang doesn't do any of this stuff.
// RUN: %clang -### %s 2>&1 | FileCheck -check-prefix=CHECK-CLANG %s
// CHECK-CLANG-NOT: "-D_DEBUG"
// CHECK-CLANG-NOT: "-D_MT"
// CHECK-CLANG-NOT: "-D_DLL"
// CHECK-CLANG-NOT: --dependent-lib

// RUN: %clang_cl -### -- %s 2>&1 | FileCheck -check-prefix=CHECK-MT %s
// RUN: %clang_cl -### /MT -- %s 2>&1 | FileCheck -check-prefix=CHECK-MT %s
// RUN: %clang -### --target=x86_64-windows-msvc -fms-runtime-lib=static -- %s \
// RUN:   2>&1 | FileCheck -check-prefix=CHECK-MT %s
// CHECK-MT-NOT: "-D_DEBUG"
// CHECK-MT: "-D_MT"
// CHECK-MT-NOT: "-D_DLL"
// CHECK-MT: "-flto-visibility-public-std"
// CHECK-MT: "--dependent-lib=libcmt"
// CHECK-MT: "--dependent-lib=oldnames"

// RUN: %clang_cl -### /MTd -- %s 2>&1 | FileCheck -check-prefix=CHECK-MTd %s
// RUN: %clang_cl -### /LD /MTd -- %s 2>&1 | FileCheck -check-prefix=CHECK-MTd %s
// RUN: %clang -### --target=x86_64-windows-msvc -fms-runtime-lib=static_dbg \
// RUN:   -- %s 2>&1 | FileCheck -check-prefix=CHECK-MTd %s
// CHECK-MTd: "-D_DEBUG"
// CHECK-MTd: "-D_MT"
// CHECK-MTd-NOT: "-D_DLL"
// CHECK-MTd: "-flto-visibility-public-std"
// CHECK-MTd: "--dependent-lib=libcmtd"
// CHECK-MTd: "--dependent-lib=oldnames"

// RUN: %clang_cl -### /MD -- %s 2>&1 | FileCheck -check-prefix=CHECK-MD %s
// RUN: %clang -### --target=x86_64-windows-msvc -fms-runtime-lib=dll -- %s \
// RUN:   2>&1 | FileCheck -check-prefix=CHECK-MD %s
// CHECK-MD-NOT: "-D_DEBUG"
// CHECK-MD: "-D_MT"
// CHECK-MD: "-D_DLL"
// CHECK-MD: "--dependent-lib=msvcrt"
// CHECK-MD: "--dependent-lib=oldnames"

// RUN: %clang_cl -### /MDd -- %s 2>&1 | FileCheck -check-prefix=CHECK-MDd %s
// RUN: %clang -### --target=x86_64-windows-msvc -fms-runtime-lib=dll_dbg -- \
// RUN:   %s 2>&1 | FileCheck -check-prefix=CHECK-MDd %s
// CHECK-MDd: "-D_DEBUG"
// CHECK-MDd: "-D_MT"
// CHECK-MDd: "-D_DLL"
// CHECK-MDd: "--dependent-lib=msvcrtd"
// CHECK-MDd: "--dependent-lib=oldnames"

// RUN: %clang_cl -### /LD -- %s 2>&1 | FileCheck -check-prefix=CHECK-LD %s
// RUN: %clang_cl -### /LD /MT -- %s 2>&1 | FileCheck -check-prefix=CHECK-LD %s
// CHECK-LD-NOT: "-D_DEBUG"
// CHECK-LD: "-D_MT"
// CHECK-LD-NOT: "-D_DLL"
// CHECK-LD: "--dependent-lib=libcmt"

// RUN: %clang_cl -### /LDd -- %s 2>&1 | FileCheck -check-prefix=CHECK-LDd %s
// RUN: %clang_cl -### /LDd /MTd -- %s 2>&1 | FileCheck -check-prefix=CHECK-LDd %s
// CHECK-LDd: "-D_DEBUG"
// CHECK-LDd: "-D_MT"
// CHECK-LDd-NOT: "-D_DLL"
// CHECK-LDd: "--dependent-lib=libcmtd"

// RUN: %clang_cl -### /LDd /MT -- %s 2>&1 | FileCheck -check-prefix=CHECK-LDdMT %s
// RUN: %clang_cl -### /MT /LDd -- %s 2>&1 | FileCheck -check-prefix=CHECK-LDdMT %s
// CHECK-LDdMT: "-D_DEBUG"
// CHECK-LDdMT: "-D_MT"
// CHECK-LDdMT-NOT: "-D_DLL"
// CHECK-LDdMT: "--dependent-lib=libcmt"

// RUN: %clang_cl -### /LD /MD -- %s 2>&1 | FileCheck -check-prefix=CHECK-LDMD %s
// RUN: %clang_cl -### /MD /LD -- %s 2>&1 | FileCheck -check-prefix=CHECK-LDMD %s
// CHECK-LDMD-NOT: "-D_DEBUG"
// CHECK-LDMD: "-D_MT"
// CHECK-LDMD: "-D_DLL"
// CHECK-LDMD: "--dependent-lib=msvcrt"

// RUN: %clang_cl -### /LDd /MD -- %s 2>&1 | FileCheck -check-prefix=CHECK-LDdMD %s
// RUN: %clang_cl -### /MD /LDd -- %s 2>&1 | FileCheck -check-prefix=CHECK-LDdMD %s
// CHECK-LDdMD: "-D_DEBUG"
// CHECK-LDdMD: "-D_MT"
// CHECK-LDdMD: "-D_DLL"
// CHECK-LDdMD: "--dependent-lib=msvcrt"

// RUN: %clang_cl -### /LD /MDd -- %s 2>&1 | FileCheck -check-prefix=CHECK-LDMDd %s
// RUN: %clang_cl -### /MDd /LD -- %s 2>&1 | FileCheck -check-prefix=CHECK-LDMDd %s
// RUN: %clang_cl -### /LDd /MDd -- %s 2>&1 | FileCheck -check-prefix=CHECK-LDMDd %s
// RUN: %clang_cl -### /MDd /LDd -- %s 2>&1 | FileCheck -check-prefix=CHECK-LDMDd %s
// CHECK-LDMDd: "-D_DEBUG"
// CHECK-LDMDd: "-D_MT"
// CHECK-LDMDd: "-D_DLL"
// CHECK-LDMDd: "--dependent-lib=msvcrtd"

// RUN: %clang_cl /MD /MT -### -- %s 2>&1 | FileCheck -check-prefix=MTOVERRIDE %s
// MTOVERRIDE: "--dependent-lib=libcmt"

// RUN: %clang_cl -### /Zl -- %s 2>&1 | FileCheck -check-prefix=CHECK-MTZl %s
// RUN: %clang_cl -### /MT /Zl -- %s 2>&1 | FileCheck -check-prefix=CHECK-MTZl %s
// RUN: %clang -### --target=x86_64-windows-msvc -fms-runtime-lib=static \
// RUN:   -fms-omit-default-lib -- %s 2>&1 | FileCheck \
// RUN:   -check-prefix=CHECK-MTZl %s
// CHECK-MTZl-NOT: "-D_DEBUG"
// CHECK-MTZl: "-D_MT"
// CHECK-MTZl-NOT: "-D_DLL"
// CHECK-MTZl-SAME: "-D_VC_NODEFAULTLIB"
// CHECK-MTZl-NOT: "--dependent-lib=libcmt"
// CHECK-MTZl-NOT: "--dependent-lib=oldnames"
