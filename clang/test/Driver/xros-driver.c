// RUN: %clang -target arm64-apple-xros1 -c -### %s 2>&1 | FileCheck --check-prefix=VERSION1 %s
// RUN: %clang -target arm64-apple-xros -c -### %s 2>&1 | FileCheck --check-prefix=VERSION1 %s

// RUN: %clang -target arm64-apple-xros1-simulator -c -### %s 2>&1 | FileCheck --check-prefix=VERSION1_ASi %s

// RUN: not %clang -target arm64-apple-xros1000 -c -### %s 2>&1 | FileCheck --check-prefix=INVALID-VERSION %s

// RUN: %clang -target arm64-apple-xros1  -### %s 2>&1 | FileCheck --check-prefix=LINK %s
// RUN: %clang -target arm64-apple-xros1-simulator  -### %s 2>&1 | FileCheck --check-prefix=LINK-SIM %s

// RUN: %clang -target arm64-apple-xros1 -c -x objective-c -fobjc-arc -### %s 2>&1 | FileCheck --check-prefix=ARC %s
// RUN: %clang -target arm64-apple-xros -c -x objective-c -fobjc-arc -### %s 2>&1 | FileCheck --check-prefix=OBJC-RUNTIME %s
// RUN: %clang -target arm64-apple-xros2 -c -x objective-c -fobjc-arc -### %s 2>&1 | FileCheck --check-prefix=OBJC-RUNTIME2 %s

// RUN: %clang -target arm64-apple-xros1 -c -x objective-c -fobjc-arc -### %s 2>&1 | FileCheck --check-prefix=ARC %s
// RUN: %clang -target arm64-apple-xros1-simulator -c -x objective-c -fobjc-arc -### %s 2>&1 | FileCheck --check-prefix=ARC %s

// RUN: %clang -target arm64-apple-xros -c -### %s 2>&1 | FileCheck --check-prefix=SSP_ON %s

// RUN: %clang -target arm64e-apple-xros -c -### %s 2>&1 | FileCheck --check-prefix=CPU-ARM64E %s
// RUN: %clang -target arm64-apple-xros -c -### %s 2>&1 | FileCheck --check-prefix=CPU-ARM64 %s
// RUN: %clang -target arm64-apple-xros-simulator -c -### %s 2>&1 | FileCheck --check-prefix=CPU-ARM64-SIM %s

// VERSION1: "-cc1"{{.*}} "-triple" "arm64-apple-xros1.0.0"
// VERSION1_ASi: "-cc1"{{.*}} "-triple" "arm64-apple-xros1.0.0-simulator"
// INVALID-VERSION: error: invalid version number in

// VERSION1-NOT: -faligned-alloc-unavailable

// LINK: "-platform_version" "xros" "1.0.0" "1.0.0"
// LINK-SIM: "-platform_version" "xros-simulator" "1.0.0" "1.0.0"

// OBJC-RUNTIME: "-fobjc-runtime=ios-17.0.0.0"
// OBJC-RUNTIME2: "-fobjc-runtime=ios-18.0.0.0"
// ARC-NOT: error:

// SSP_ON: "-stack-protector" "1"

// CPU-ARM64E: "-cc1"{{.*}} "-triple" "arm64e-apple-xros1.0.0"{{.*}} "-target-cpu" "apple-a12"{{.*}}
// CPU-ARM64: "-cc1"{{.*}} "-triple" "arm64-apple-xros1.0.0"{{.*}} "-target-cpu" "apple-a12"
// CPU-ARM64-SIM: "-cc1"{{.*}} "-triple" "arm64-apple-xros1.0.0-simulator"{{.*}} "-target-cpu" "apple-m1"
