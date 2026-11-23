// RUN: %clang_cc1 -E -dM -triple=arm64ec-windows-msvc -std=c89 -fms-compatibility-version=19.33 -ffreestanding < /dev/null | FileCheck -check-prefix=C89_MSVC33 %s
// RUN: %clang_cc1 -E -dM -triple=arm64ec-windows-msvc -std=c99 -fms-compatibility-version=19.33 -ffreestanding < /dev/null | FileCheck -check-prefix=C99_MSVC33 %s
// RUN: %clang_cc1 -E -dM -triple=arm64ec-windows-msvc -std=c11 -fms-compatibility-version=19.33 -ffreestanding < /dev/null | FileCheck -check-prefix=C11_MSVC33 %s
// RUN: %clang_cc1 -E -dM -triple=arm64ec-windows-msvc -std=c89 -fms-compatibility-version=19.39 -ffreestanding < /dev/null | FileCheck -check-prefix=C89_MSVC39 %s
// RUN: %clang_cc1 -E -dM -triple=arm64ec-windows-msvc -std=c99 -fms-compatibility-version=19.39 -ffreestanding < /dev/null | FileCheck -check-prefix=C99_MSVC39 %s
// RUN: %clang_cc1 -E -dM -triple=arm64ec-windows-msvc -std=c11 -fms-compatibility-version=19.39 -ffreestanding < /dev/null | FileCheck -check-prefix=C11_MSVC39 %s
// RUN: %clang_cc1 -E -dM -triple=arm64ec-windows-msvc -std=c11 -fms-compatibility-version=19.40 -ffreestanding < /dev/null | FileCheck -check-prefix=C11_MSVC40 %s

// C89_MSVC33: #define __STDC_NO_THREADS__ 1
// C99_MSVC33: #define __STDC_NO_THREADS__ 1
// C11_MSVC33: #define __STDC_NO_THREADS__ 1
// C89_MSVC39: #define __STDC_NO_THREADS__ 1
// C99_MSVC39: #define __STDC_NO_THREADS__ 1
// C11_MSVC39-NOT: #define __STDC_NO_THREADS__
// C11_MSVC40-NOT: #define __STDC_NO_THREADS__
