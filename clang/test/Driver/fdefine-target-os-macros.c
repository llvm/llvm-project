// RUN: %clang -### --target=arm64-apple-darwin %s 2>&1 | FileCheck %s --check-prefix=DARWIN-DEFAULT
// DARWIN-DEFAULT-NOT: "-fdefine-target-os-macros"

// RUN: %clang -### --target=arm-none-linux-gnu %s 2>&1 | FileCheck %s --check-prefix=NON-DARWIN-DEFAULT
// RUN: %clang -### --target=x86_64-pc-win32 %s 2>&1 | FileCheck %s --check-prefix=NON-DARWIN-DEFAULT
// NON-DARWIN-DEFAULT-NOT: "-fdefine-target-os-macros"

// RUN: %clang -dM -E --target=arm64-apple-macos \
// RUN:        -fdefine-target-os-macros %s 2>&1 \
// RUN: | FileCheck %s -DMAC=1         \
// RUN:                -DOSX=1         \
// RUN:                -DIPHONE=0      \
// RUN:                -DIOS=0         \
// RUN:                -DTV=0          \
// RUN:                -DWATCH=0       \
// RUN:                -DVISION=0      \
// RUN:                -DDRIVERKIT=0   \
// RUN:                -DMACCATALYST=0 \
// RUN:                -DEMBEDDED=0    \
// RUN:                -DSIMULATOR=0   \
// RUN:                -DWINDOWS=0     \
// RUN:                -DLINUX=0       \
// RUN:                -DUNIX=0

// RUN: %clang -dM -E --target=arm64-apple-ios \
// RUN:        -fdefine-target-os-macros %s 2>&1 \
// RUN: | FileCheck %s -DMAC=1         \
// RUN:                -DOSX=0         \
// RUN:                -DIPHONE=1      \
// RUN:                -DIOS=1         \
// RUN:                -DTV=0          \
// RUN:                -DWATCH=0       \
// RUN:                -DVISION=0      \
// RUN:                -DDRIVERKIT=0   \
// RUN:                -DMACCATALYST=0 \
// RUN:                -DEMBEDDED=1    \
// RUN:                -DSIMULATOR=0   \
// RUN:                -DWINDOWS=0     \
// RUN:                -DLINUX=0       \
// RUN:                -DUNIX=0

// RUN: %clang -dM -E --target=arm64-apple-ios-macabi \
// RUN:        -fdefine-target-os-macros %s 2>&1 \
// RUN: | FileCheck %s -DMAC=1         \
// RUN:                -DOSX=0         \
// RUN:                -DIPHONE=1      \
// RUN:                -DIOS=1         \
// RUN:                -DTV=0          \
// RUN:                -DWATCH=0       \
// RUN:                -DVISION=0      \
// RUN:                -DDRIVERKIT=0   \
// RUN:                -DMACCATALYST=1 \
// RUN:                -DEMBEDDED=0    \
// RUN:                -DSIMULATOR=0   \
// RUN:                -DWINDOWS=0     \
// RUN:                -DLINUX=0       \
// RUN:                -DUNIX=0

// RUN: %clang -dM -E --target=arm64-apple-ios-simulator \
// RUN:        -fdefine-target-os-macros %s 2>&1 \
// RUN: | FileCheck %s -DMAC=1         \
// RUN:                -DOSX=0         \
// RUN:                -DIPHONE=1      \
// RUN:                -DIOS=1         \
// RUN:                -DTV=0          \
// RUN:                -DWATCH=0       \
// RUN:                -DVISION=0      \
// RUN:                -DDRIVERKIT=0   \
// RUN:                -DMACCATALYST=0 \
// RUN:                -DEMBEDDED=0    \
// RUN:                -DSIMULATOR=1   \
// RUN:                -DWINDOWS=0     \
// RUN:                -DLINUX=0       \
// RUN:                -DUNIX=0

// RUN: %clang -dM -E --target=arm64-apple-tvos \
// RUN:        -fdefine-target-os-macros %s 2>&1 \
// RUN: | FileCheck %s -DMAC=1         \
// RUN:                -DOSX=0         \
// RUN:                -DIPHONE=1      \
// RUN:                -DIOS=0         \
// RUN:                -DTV=1          \
// RUN:                -DWATCH=0       \
// RUN:                -DVISION=0      \
// RUN:                -DDRIVERKIT=0   \
// RUN:                -DMACCATALYST=0 \
// RUN:                -DEMBEDDED=1    \
// RUN:                -DSIMULATOR=0   \
// RUN:                -DWINDOWS=0     \
// RUN:                -DLINUX=0       \
// RUN:                -DUNIX=0

// RUN: %clang -dM -E --target=arm64-apple-tvos-simulator \
// RUN:        -fdefine-target-os-macros %s 2>&1 \
// RUN: | FileCheck %s -DMAC=1         \
// RUN:                -DOSX=0         \
// RUN:                -DIPHONE=1      \
// RUN:                -DIOS=0         \
// RUN:                -DTV=1          \
// RUN:                -DWATCH=0       \
// RUN:                -DVISION=0      \
// RUN:                -DDRIVERKIT=0   \
// RUN:                -DMACCATALYST=0 \
// RUN:                -DEMBEDDED=0    \
// RUN:                -DSIMULATOR=1   \
// RUN:                -DWINDOWS=0     \
// RUN:                -DLINUX=0       \
// RUN:                -DUNIX=0

// RUN: %clang -dM -E --target=arm64-apple-watchos \
// RUN:        -fdefine-target-os-macros %s 2>&1 \
// RUN: | FileCheck %s -DMAC=1         \
// RUN:                -DOSX=0         \
// RUN:                -DIPHONE=1      \
// RUN:                -DIOS=0         \
// RUN:                -DTV=0          \
// RUN:                -DWATCH=1       \
// RUN:                -DVISION=0      \
// RUN:                -DDRIVERKIT=0   \
// RUN:                -DMACCATALYST=0 \
// RUN:                -DEMBEDDED=1    \
// RUN:                -DSIMULATOR=0   \
// RUN:                -DWINDOWS=0     \
// RUN:                -DLINUX=0       \
// RUN:                -DUNIX=0

// RUN: %clang -dM -E --target=arm64-apple-watchos-simulator \
// RUN:        -fdefine-target-os-macros %s 2>&1 \
// RUN: | FileCheck %s -DMAC=1         \
// RUN:                -DOSX=0         \
// RUN:                -DIPHONE=1      \
// RUN:                -DIOS=0         \
// RUN:                -DTV=0          \
// RUN:                -DWATCH=1       \
// RUN:                -DVISION=0      \
// RUN:                -DDRIVERKIT=0   \
// RUN:                -DMACCATALYST=0 \
// RUN:                -DEMBEDDED=0    \
// RUN:                -DSIMULATOR=1   \
// RUN:                -DWINDOWS=0     \
// RUN:                -DLINUX=0       \
// RUN:                -DUNIX=0

// RUN: %clang -dM -E --target=arm64-apple-xros %s 2>&1 \
// RUN: | FileCheck %s -DMAC=1         \
// RUN:                -DOSX=0         \
// RUN:                -DIPHONE=1      \
// RUN:                -DIOS=0         \
// RUN:                -DTV=0          \
// RUN:                -DWATCH=0       \
// RUN:                -DVISION=1      \
// RUN:                -DDRIVERKIT=0   \
// RUN:                -DMACCATALYST=0 \
// RUN:                -DEMBEDDED=1    \
// RUN:                -DSIMULATOR=0   \
// RUN:                -DWINDOWS=0     \
// RUN:                -DLINUX=0       \
// RUN:                -DUNIX=0

// RUN: %clang -dM -E --target=arm64-apple-xros-simulator %s 2>&1 \
// RUN: | FileCheck %s -DMAC=1         \
// RUN:                -DOSX=0         \
// RUN:                -DIPHONE=1      \
// RUN:                -DIOS=0         \
// RUN:                -DTV=0          \
// RUN:                -DWATCH=0       \
// RUN:                -DVISION=1      \
// RUN:                -DDRIVERKIT=0   \
// RUN:                -DMACCATALYST=0 \
// RUN:                -DEMBEDDED=0    \
// RUN:                -DSIMULATOR=1   \
// RUN:                -DWINDOWS=0     \
// RUN:                -DLINUX=0       \
// RUN:                -DUNIX=0

// RUN: %clang -dM -E --target=arm64-apple-driverkit \
// RUN:        -fdefine-target-os-macros %s 2>&1 \
// RUN: | FileCheck %s -DMAC=1         \
// RUN:                -DOSX=0         \
// RUN:                -DIPHONE=0      \
// RUN:                -DIOS=0         \
// RUN:                -DTV=0          \
// RUN:                -DWATCH=0       \
// RUN:                -DVISION=0      \
// RUN:                -DDRIVERKIT=1   \
// RUN:                -DMACCATALYST=0 \
// RUN:                -DEMBEDDED=0    \
// RUN:                -DSIMULATOR=0   \
// RUN:                -DWINDOWS=0     \
// RUN:                -DLINUX=0       \
// RUN:                -DUNIX=0

// RUN: %clang -dM -E --target=x86_64-pc-linux-gnu \
// RUN:        -fdefine-target-os-macros %s 2>&1 \
// RUN: | FileCheck %s -DMAC=0         \
// RUN:                -DOSX=0         \
// RUN:                -DIPHONE=0      \
// RUN:                -DIOS=0         \
// RUN:                -DTV=0          \
// RUN:                -DWATCH=0       \
// RUN:                -DVISION=0      \
// RUN:                -DDRIVERKIT=0   \
// RUN:                -DMACCATALYST=0 \
// RUN:                -DEMBEDDED=0    \
// RUN:                -DSIMULATOR=0   \
// RUN:                -DWINDOWS=0     \
// RUN:                -DLINUX=1       \
// RUN:                -DUNIX=0

// RUN: %clang -dM -E --target=x86_64-pc-win32 \
// RUN:        -fdefine-target-os-macros %s 2>&1 \
// RUN: | FileCheck %s -DMAC=0         \
// RUN:                -DOSX=0         \
// RUN:                -DIPHONE=0      \
// RUN:                -DIOS=0         \
// RUN:                -DTV=0          \
// RUN:                -DWATCH=0       \
// RUN:                -DVISION=0      \
// RUN:                -DDRIVERKIT=0   \
// RUN:                -DMACCATALYST=0 \
// RUN:                -DEMBEDDED=0    \
// RUN:                -DSIMULATOR=0   \
// RUN:                -DWINDOWS=1     \
// RUN:                -DLINUX=0       \
// RUN:                -DUNIX=0

// RUN: %clang -dM -E --target=x86_64-pc-windows-gnu \
// RUN:        -fdefine-target-os-macros %s 2>&1 \
// RUN: | FileCheck %s -DMAC=0         \
// RUN:                -DOSX=0         \
// RUN:                -DIPHONE=0      \
// RUN:                -DIOS=0         \
// RUN:                -DTV=0          \
// RUN:                -DWATCH=0       \
// RUN:                -DVISION=0      \
// RUN:                -DDRIVERKIT=0   \
// RUN:                -DMACCATALYST=0 \
// RUN:                -DEMBEDDED=0    \
// RUN:                -DSIMULATOR=0   \
// RUN:                -DWINDOWS=1     \
// RUN:                -DLINUX=0       \
// RUN:                -DUNIX=0

// RUN: %clang -dM -E --target=sparc-none-solaris \
// RUN:        -fdefine-target-os-macros %s 2>&1 \
// RUN: | FileCheck %s -DMAC=0         \
// RUN:                -DOSX=0         \
// RUN:                -DIPHONE=0      \
// RUN:                -DIOS=0         \
// RUN:                -DTV=0          \
// RUN:                -DWATCH=0       \
// RUN:                -DVISION=0      \
// RUN:                -DDRIVERKIT=0   \
// RUN:                -DMACCATALYST=0 \
// RUN:                -DEMBEDDED=0    \
// RUN:                -DSIMULATOR=0   \
// RUN:                -DWINDOWS=0     \
// RUN:                -DLINUX=0       \
// RUN:                -DUNIX=1

// RUN: %clang -dM -E --target=arm64-apple-macos \
// RUN:        -fno-define-target-os-macros %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=NEG

// RUN: %clang -dM -E --target=arm64-apple-macos \
// RUN:        -fdefine-target-os-macros \
// RUN:        -fno-define-target-os-macros %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=NEG

// RUN: %clang -dM -E --target=x86_64-pc-windows \
// RUN:        -fdefine-target-os-macros \
// RUN:        -fno-define-target-os-macros %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=NEG

// NEG-NOT: #define TARGET_OS_

// CHECK-DAG: #define TARGET_OS_MAC [[MAC]]
// CHECK-DAG: #define TARGET_OS_OSX [[OSX]]
// CHECK-DAG: #define TARGET_OS_IPHONE [[IPHONE]]
// CHECK-DAG: #define TARGET_OS_IOS [[IOS]]
// CHECK-DAG: #define TARGET_OS_TV [[TV]]
// CHECK-DAG: #define TARGET_OS_WATCH [[WATCH]]
// CHECK-DAG: #define TARGET_OS_VISION [[VISION]]
// CHECK-DAG: #define TARGET_OS_DRIVERKIT [[DRIVERKIT]]
// CHECK-DAG: #define TARGET_OS_MACCATALYST [[MACCATALYST]]
// CHECK-DAG: #define TARGET_OS_SIMULATOR [[SIMULATOR]]
// Deprecated
// CHECK-DAG: #define TARGET_OS_EMBEDDED [[EMBEDDED]]
// CHECK-DAG: #define TARGET_OS_NANO [[WATCH]]
// CHECK-DAG: #define TARGET_IPHONE_SIMULATOR [[SIMULATOR]]
// CHECK-DAG: #define TARGET_OS_UIKITFORMAC [[MACCATALYST]]
// Non-darwin OSes
// CHECK-DAG: #define TARGET_OS_WIN32 [[WINDOWS]]
// CHECK-DAG: #define TARGET_OS_WINDOWS [[WINDOWS]]
// CHECK-DAG: #define TARGET_OS_LINUX [[LINUX]]
// CHECK-DAG: #define TARGET_OS_UNIX [[UNIX]]
