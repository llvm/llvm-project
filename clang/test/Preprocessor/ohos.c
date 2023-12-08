// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding -triple=arm-linux-ohos < /dev/null | FileCheck %s -match-full-lines -check-prefix=ARM-OHOS-CXX
// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding -triple=aarch64-linux-ohos < /dev/null | FileCheck %s -match-full-lines -check-prefix=ARM64-OHOS-CXX
// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding -triple=riscv64-linux-ohos < /dev/null | FileCheck %s -match-full-lines -check-prefix=RISCV64-OHOS-CXX
// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding -triple=mipsel-linux-ohos < /dev/null | FileCheck %s -match-full-lines -check-prefix=MIPSEL-OHOS-CXX
// RUN: %clang_cc1 -x c++ -E -dM -ffreestanding -triple=x86_64-linux-ohos < /dev/null | FileCheck %s -match-full-lines -check-prefix=X86_64-OHOS-CXX
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=arm-linux-ohos < /dev/null | FileCheck %s -check-prefix=OHOS-DEFS

// ARM-OHOS-CXX: #define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 8U
// ARM64-OHOS-CXX: #define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 16UL
// RISCV64-OHOS-CXX: #define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 16UL
// MIPSEL-OHOS-CXX: #define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 8U
// X86_64-OHOS-CXX: #define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 16UL
// OHOS-DEFS: __OHOS_FAMILY__
// OHOS-DEFS: __OHOS__
// OHOS-DEFS-NOT: __OHOS__
