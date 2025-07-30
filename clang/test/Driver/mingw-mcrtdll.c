// RUN: %clang -v --target=x86_64-w64-mingw32 -### %s 2>&1 | FileCheck -check-prefix=DEFAULT %s
// RUN: %clang -v --target=x86_64-w64-mingw32 -mcrtdll=msvcr90 -### %s 2>&1 | FileCheck -check-prefix=MSVCR90 %s
// RUN: %clang -v --target=x86_64-w64-mingw32 -mcrtdll=msvcr90_suffix -### %s 2>&1 | FileCheck -check-prefix=MSVCR90_SUFFIX %s
// RUN: %clang -v --target=x86_64-w64-mingw32 -mcrtdll=ucrt -### %s 2>&1 | FileCheck -check-prefix=UCRT %s
// RUN: %clang -v --target=x86_64-w64-mingw32 -mcrtdll=ucrtbase -### %s 2>&1 | FileCheck -check-prefix=UCRTBASE %s

// RUN: %clang -dM -E --target=x86_64-w64-mingw32 %s 2>&1  | FileCheck -check-prefix=DEFINE_DEFAULT %s
// RUN: %clang -dM -E --target=x86_64-w64-mingw32 -mcrtdll=msvcr90 %s 2>&1  | FileCheck -check-prefix=DEFINE_MSVCR90 %s
// RUN: %clang -dM -E --target=x86_64-w64-mingw32 -mcrtdll=msvcr90_suffix %s 2>&1  | FileCheck -check-prefix=DEFINE_MSVCR90 %s
// RUN: %clang -dM -E --target=x86_64-w64-mingw32 -mcrtdll=ucrt %s 2>&1  | FileCheck -check-prefix=DEFINE_UCRT %s
// RUN: %clang -dM -E --target=x86_64-w64-mingw32 -mcrtdll=ucrtbase %s 2>&1  | FileCheck -check-prefix=DEFINE_UCRT %s
// RUN: not %clang -dM -E --target=x86_64-w64-mingw32 -mcrtdll=bad %s 2>&1  | FileCheck -check-prefix=BAD %s

// DEFAULT: "-lmingwex" "-lmsvcrt"
// DEFINE_DEFAULT: #define __MSVCRT__
// MSVCR90: "-lmingwex" "-lmsvcr90"
// DEFINE_MSVCR90: #define __MSVCRT_VERSION__ 0x900
// DEFINE_MSVCR90: #define __MSVCRT__
// MSVCR90-NOT: "-lmsvcrt"
// MSVCR90_SUFFIX: "-lmingwex" "-lmsvcr90_suffix"
// MSVCR90_SUFFIX-NOT: "-lmsvcrt"
// UCRT: "-lmingwex" "-lucrt"
// DEFINE_UCRT: #define _UCRT
// DEFINE_UCRT-NOT: #define __MSVCRT_VERSION__
// UCRT-NOT: "-lmsvcrt"
// UCRTBASE: "-lmingwex" "-lucrtbase"
// UCRTBASE-NOT: "-lmsvcrt"
// DEFINE_CRTDLL: #define __CRTDLL__
// DEFINE_CRTDLL-NOT: #define __MSVCRT__
// BAD: error: unknown Windows/MinGW C runtime library 'bad'
