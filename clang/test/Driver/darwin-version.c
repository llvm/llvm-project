// RUN: %clang -target armv6-apple-darwin9 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX %s
// CHECK-VERSION-OSX: "armv6k-apple-macosx10.5.0"
// RUN: %clang -target armv6-apple-darwin9 -miphoneos-version-min=2.0 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-IOS2 %s
// CHECK-VERSION-IOS2: "armv6k-apple-ios2.0.0"
// RUN: %clang -target armv6-apple-darwin9 -miphoneos-version-min=2.2 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-IOS22 %s
// CHECK-VERSION-IOS22: "armv6k-apple-ios2.2.0"
// RUN: %clang -target armv6-apple-darwin9 -miphoneos-version-min=3.0 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-IOS3 %s
// CHECK-VERSION-IOS3: "armv6k-apple-ios3.0.0"

// RUN: env IPHONEOS_DEPLOYMENT_TARGET=11.0 \
// RUN:   not %clang -target armv7-apple-darwin -c -### %s 2> %t.err
// RUN:   FileCheck --input-file=%t.err --check-prefix=CHECK-VERSION-IOS4 %s
// CHECK-VERSION-IOS4: invalid iOS deployment version 'IPHONEOS_DEPLOYMENT_TARGET=11.0'

// RUN: not %clang -target armv7-apple-ios11.0 -c -### %s 2> %t.err
// RUN: FileCheck --input-file=%t.err --check-prefix=CHECK-VERSION-IOS41 %s
// CHECK-VERSION-IOS41: invalid iOS deployment version '-target armv7-apple-ios11.0'

// RUN: not %clang -target armv7-apple-darwin -miphoneos-version-min=11.0 -c -### %s 2> %t.err
// RUN: FileCheck --input-file=%t.err --check-prefix=CHECK-VERSION-IOS5 %s
// CHECK-VERSION-IOS5: invalid iOS deployment version '-miphoneos-version-min=11.0'

// RUN: not %clang -target i386-apple-darwin -mios-simulator-version-min=11.0 -c -### %s 2> %t.err
// RUN: FileCheck --input-file=%t.err --check-prefix=CHECK-VERSION-IOS6 %s
// CHECK-VERSION-IOS6: invalid iOS deployment version '-mios-simulator-version-min=11.0'

// RUN: not %clang -target armv7-apple-ios11.1 -c -### %s 2>&1 | \
// RUN: FileCheck --check-prefix=CHECK-VERSION-IOS71 %s
// CHECK-VERSION-IOS71: invalid iOS deployment version
// RUN: %clang -target armv7-apple-darwin -Wno-missing-sysroot -isysroot SDKs/iPhoneOS11.0.sdk -c -### %s 2>&1 | \
// RUN: FileCheck --check-prefix=CHECK-VERSION-IOS7 %s
// CHECK-VERSION-IOS7: thumbv7-apple-ios10.99.99

// RUN: env IPHONEOS_DEPLOYMENT_TARGET=11.0 \
// RUN:   %clang -target arm64-apple-darwin -c -### %s 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-IOS8 %s
// CHECK-VERSION-IOS8: arm64-apple-ios11.0.0

// RUN: %clang -target arm64-apple-ios11.0 -miphoneos-version-min=11.0 -c -### %s 2>&1 | \
// RUN: FileCheck --check-prefix=CHECK-VERSION-IOS9 %s
// CHECK-VERSION-IOS9: arm64-apple-ios11.0.0

// RUN: %clang -target x86_64-apple-darwin -mios-simulator-version-min=11.0 -c -### %s 2>&1 | \
// RUN: FileCheck --check-prefix=CHECK-VERSION-IOS10 %s
// CHECK-VERSION-IOS10: x86_64-apple-ios11.0.0-simulator

// RUN: %clang -target arm64-apple-darwin -mios-simulator-version-min=11.0 -c -### %s 2>&1 | \
// RUN: FileCheck --check-prefix=CHECK-VERSION-IOS10-ARM64 %s
// CHECK-VERSION-IOS10-ARM64: arm64-apple-ios11.0.0-simulator

// RUN: %clang -target arm64-apple-ios11.1 -c -### %s 2>&1 | \
// RUN: FileCheck --check-prefix=CHECK-VERSION-IOS11 %s
// CHECK-VERSION-IOS11: arm64-apple-ios11.1.0

// RUN: %clang -target armv7-apple-ios9.0 -miphoneos-version-min=11.0 -c -Wno-invalid-ios-deployment-target -### %s 2>&1 | \
// RUN: FileCheck --check-prefix=CHECK-VERSION-IOS12 %s
// CHECK-VERSION-IOS12: thumbv7-apple-ios9.0.0

// RUN: %clang -target i686-apple-darwin8 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX4 %s
// RUN: %clang -target i686-apple-darwin9 -mmacosx-version-min=10.4 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX4 %s
// CHECK-VERSION-OSX4: "i686-apple-macosx10.4.0"
// RUN: %clang -target i686-apple-darwin9 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX5 %s
// RUN: %clang -target i686-apple-darwin9 -mmacosx-version-min=10.5 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX5 %s
// CHECK-VERSION-OSX5: "i686-apple-macosx10.5.0"
// RUN: %clang -target i686-apple-darwin10 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX6 %s
// RUN: %clang -target i686-apple-darwin9 -mmacosx-version-min=10.6 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX6 %s
// CHECK-VERSION-OSX6: "i686-apple-macosx10.6.0"
// RUN: %clang -target x86_64-apple-darwin14 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX10 %s
// RUN: %clang -target x86_64-apple-darwin -mmacosx-version-min=10.10 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX10 %s
// RUN: %clang -target x86_64-apple-darwin -mmacos-version-min=10.10 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX10 %s
// CHECK-VERSION-OSX10: "x86_64-apple-macosx10.10.0"
// RUN: not %clang -target x86_64-apple-darwin -mmacosx-version-min= -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-MISSING %s
// RUN: not %clang -target x86_64-apple-darwin -mmacos-version-min= -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-MISSING %s
// CHECK-VERSION-MISSING: invalid version number
// RUN: %clang -target armv7k-apple-darwin -mwatchos-version-min=2.0 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-WATCHOS20 %s
// RUN: %clang -target armv7-apple-darwin -mtvos-version-min=8.3 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-TVOS83 %s
// CHECK-VERSION-TVOS83: "thumbv7-apple-tvos8.3.0"
// RUN: %clang -target i386-apple-darwin -mtvos-simulator-version-min=8.3 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-TVSIM83 %s
// CHECK-VERSION-TVSIM83: "i386-apple-tvos8.3.0-simulator"
// RUN: %clang -target armv7k-apple-darwin -mwatchos-version-min=2.0 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-WATCHOS20 %s
// CHECK-VERSION-WATCHOS20: "thumbv7k-apple-watchos2.0.0"
// RUN: %clang -target i386-apple-darwin -mwatchos-simulator-version-min=2.0 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-WATCHSIM20 %s
// CHECK-VERSION-WATCHSIM20: "i386-apple-watchos2.0.0-simulator"

// RUN: %clang -target x86_64-apple-driverkit19.0 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-DRIVERKIT190 %s
// CHECK-VERSION-DRIVERKIT190: "x86_64-apple-driverkit19.0.0"

// Check environment variable gets interpreted correctly
// RUN: env MACOSX_DEPLOYMENT_TARGET=10.5 IPHONEOS_DEPLOYMENT_TARGET=2.0 \
// RUN:   %clang -target i686-apple-darwin9 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX5 %s
// RUN: env MACOSX_DEPLOYMENT_TARGET=10.5 IPHONEOS_DEPLOYMENT_TARGET=2.0 \
// RUN:   %clang -target armv6-apple-darwin9 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-IOS2 %s

// RUN: env MACOSX_DEPLOYMENT_TARGET=10.4.10 \
// RUN:   %clang -target i386-apple-darwin9 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX49 %s
// CHECK-VERSION-OSX49: "i386-apple-macosx10.4.10"
// RUN: env IPHONEOS_DEPLOYMENT_TARGET=2.3.1 \
// RUN:   %clang -target armv6-apple-darwin9 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-IOS231 %s
// CHECK-VERSION-IOS231: "armv6k-apple-ios2.3.1"

// RUN: env MACOSX_DEPLOYMENT_TARGET=10.5 TVOS_DEPLOYMENT_TARGET=8.3.1 \
// RUN:   %clang -target armv7-apple-darwin9 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-TVOS %s
// CHECK-VERSION-TVOS: "thumbv7-apple-tvos8.3.1"
// RUN: env TVOS_DEPLOYMENT_TARGET=8.3.1 \
// RUN:   %clang -target i386-apple-darwin9 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-TVOSSIM %s
// CHECK-VERSION-TVOSSIM: "i386-apple-tvos8.3.1-simulator"

// RUN: env MACOSX_DEPLOYMENT_TARGET=10.5 WATCHOS_DEPLOYMENT_TARGET=2.0 \
// RUN:   %clang -target armv7-apple-darwin9 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-WATCHOS %s
// CHECK-VERSION-WATCHOS: "thumbv7-apple-watchos2.0.0"
// RUN: env WATCHOS_DEPLOYMENT_TARGET=2.0 \
// RUN:   %clang -target i386-apple-darwin9 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-WATCHOSSIM %s
// CHECK-VERSION-WATCHOSSIM: "i386-apple-watchos2.0.0-simulator"

// RUN: %clang -target x86_64-apple-ios11.0.0 -c %s -### 2>&1 | \
// RUN: FileCheck --check-prefix=CHECK-VERSION-IOS-TARGET %s
// CHECK-VERSION-IOS-TARGET: "x86_64-apple-ios11.0.0-simulator"

// RUN: %clang -target x86_64-apple-tvos11.0 -c %s -### 2>&1 | \
// RUN: FileCheck --check-prefix=CHECK-VERSION-TVOS-TARGET %s
// CHECK-VERSION-TVOS-TARGET: "x86_64-apple-tvos11.0.0-simulator"

// RUN: %clang -target x86_64-apple-watchos4.0 -c %s -### 2>&1 | \
// RUN: FileCheck --check-prefix=CHECK-VERSION-WATCHOS-TARGET %s
// CHECK-VERSION-WATCHOS-TARGET: "x86_64-apple-watchos4.0.0-simulator"

// RUN: env DRIVERKIT_DEPLOYMENT_TARGET=19.0 \
// RUN:   %clang -target x86_64-apple-darwin -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-DRIVERKIT %s
// CHECK-VERSION-DRIVERKIT: "x86_64-apple-driverkit19.0.0"
//
// Make sure stdlib is not mistaken
// RUN: env DRIVERKIT_DEPLOYMENT_TARGET=2.0 \
// RUN:   not %clang -target arm64-apple-darwin -c -x c++ %s -stdlib=libc++ -### 2>&1

// RUN: env MACOSX_DEPLOYMENT_TARGET=1000.1000 \
// RUN:   not %clang -target x86_64-apple-darwin -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-INVALID-ENV %s
// CHECK-VERSION-INVALID-ENV: invalid version number in 'MACOSX_DEPLOYMENT_TARGET=1000.1000'



// Target can specify the OS version:

// RUN: %clang -target x86_64-apple-macos10.11.2 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-TMAC2 %s
// CHECK-VERSION-TMAC2: "x86_64-apple-macosx10.11.2"

// RUN: %clang -target arm64-apple-ios11.1 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-TIOS1 %s
// CHECK-VERSION-TIOS1: "arm64-apple-ios11.1.0"

// RUN: %clang -target arm64-apple-tvos10.3 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-TTVOS1 %s
// CHECK-VERSION-TTVOS1: "arm64-apple-tvos10.3.0"

// RUN: %clang -target armv7k-apple-watchos4.1 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-TWATCHOS1 %s
// CHECK-VERSION-TWATCHOS1: "thumbv7k-apple-watchos4.1.0"

// "darwin" always back to the -m<os>version-min and environment:

// RUN: %clang -target x86_64-apple-darwin14 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-TDARWIN-FALL1 %s
// CHECK-VERSION-TDARWIN-FALL1: "x86_64-apple-macosx10.10.0"

// RUN: %clang -target x86_64-apple-darwin14 -miphoneos-version-min=10.1 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-TDARWIN-FALL2 %s
// CHECK-VERSION-TDARWIN-FALL2: "x86_64-apple-ios10.1.0-simulator"

// RUN: env IPHONEOS_DEPLOYMENT_TARGET=9.1 \
// RUN:   %clang -target arm64-apple-darwin14 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-TDARWIN-FALL3 %s
// CHECK-VERSION-TDARWIN-FALL3: "arm64-apple-ios9.1.0"

// RUN: %clang -target arm64-apple-darwin14 -isysroot SDKs/iPhoneOS11.0.sdk -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-TDARWIN-FALL4 %s
// CHECK-VERSION-TDARWIN-FALL4: "arm64-apple-ios11.0.0"

// RUN: %clang -target unknown-apple-darwin12  -arch armv7  -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-TDARWIN-FALL5 %s
// CHECK-VERSION-TDARWIN-FALL5: "thumbv7-apple-ios5.0.0"

// Warn about -m<os>-version-min when it's used with target:

// RUN: %clang -target x86_64-apple-macos10.11.2 -mmacos-version-min=10.6 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-TNO-OSV1 %s
// CHECK-VERSION-TNO-OSV1: overriding '-mmacos-version-min=10.6' option with '-target x86_64-apple-macos10.11.2'

// RUN: %clang -target x86_64-apple-macos -miphoneos-version-min=9.1 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-TNO-OSV2 %s
// CHECK-VERSION-TNO-OSV2: overriding '-miphoneos-version-min=9.1' option with '-target x86_64-apple-macos'

// RUN: %clang -target x86_64-apple-ios -miphonesimulator-version-min=10.0 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-TNO-OSV3 %s
// CHECK-VERSION-TNO-OSV3: "x86_64-apple-ios10.0.0-simulator"
// CHECK-VERSION-TNO-OSV3-NOT: overriding '-mios-simulator-version-min
// CHECK-VERSION-TNO-OSV3-NOT: argument unused during compilation

// RUN: %clang -target arm64-apple-ios10.1.0 -miphoneos-version-min=10.1.0.1 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-TNO-OSV4 %s
// CHECK-VERSION-TNO-OSV4: overriding '-miphoneos-version-min=10.1.0.1' option with '-target arm64-apple-ios10.1.0'

// RUN: %clang -target x86_64-apple-macos10.6 -mmacos-version-min=10.6 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-TNO-SAME %s
// CHECK-VERSION-TNO-SAME-NOT: overriding
// CHECK-VERSION-TNO-SAME-NOT: argument unused during compilation

// Target with OS version is not overridden by -m<os>-version-min variables:

// RUN: %clang -target x86_64-apple-macos10.11.2 -mmacos-version-min=10.6 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-TIGNORE-OSV1 %s
// CHECK-VERSION-TIGNORE-OSV1: "x86_64-apple-macosx10.11.2"

// RUN: %clang -target arm64-apple-ios11.0 -mios-version-min=9.0 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-TIGNORE-OSV2 %s
// CHECK-VERSION-TIGNORE-OSV2: "arm64-apple-ios11.0.0"

// RUN: %clang -target arm64-apple-tvos11.0 -mtvos-version-min=9.0 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-TIGNORE-OSV3 %s
// CHECK-VERSION-TIGNORE-OSV3: "arm64-apple-tvos11.0.0"

// RUN: %clang -target armv7k-apple-watchos3 -mwatchos-version-min=4 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-TIGNORE-OSV4 %s
// CHECK-VERSION-TIGNORE-OSV4: "thumbv7k-apple-watchos3.0.0"

// Target without OS version includes the OS given by -m<os>-version-min arguments:

// RUN: %clang -target x86_64-apple-macos -mmacos-version-min=10.11 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-USE-OS-ARG1 %s
// CHECK-VERSION-USE-OS-ARG1: "x86_64-apple-macosx10.11.0"

// RUN: %clang -target arm64-apple-ios -mios-version-min=9.0 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-USE-OS-ARG2 %s
// CHECK-VERSION-USE-OS-ARG2: "arm64-apple-ios9.0.0"

// RUN: %clang -target arm64-apple-tvos -mtvos-version-min=10.0 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-USE-OS-ARG3 %s
// CHECK-VERSION-USE-OS-ARG3: "arm64-apple-tvos10.0.0"

// RUN: %clang -target armv7k-apple-watchos -mwatchos-version-min=4 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-USE-OS-ARG4 %s
// CHECK-VERSION-USE-OS-ARG4: "thumbv7k-apple-watchos4.0.0"

// Target with OS version is not overridden by environment variables:

// RUN: env MACOSX_DEPLOYMENT_TARGET=10.1 \
// RUN:   %clang -target i386-apple-macos10.5 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-TMACOS-CMD %s
// CHECK-VERSION-TMACOS-CMD: "i386-apple-macosx10.5.0"

// RUN: env IPHONEOS_DEPLOYMENT_TARGET=10.1 \
// RUN:   %clang -target arm64-apple-ios11 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-TIOS-CMD %s
// CHECK-VERSION-TIOS-CMD: "arm64-apple-ios11.0.0"

// RUN: env TVOS_DEPLOYMENT_TARGET=8.3.1 \
// RUN:   %clang -target arm64-apple-tvos9 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-TTVOS-CMD %s
// CHECK-VERSION-TTVOS-CMD: "arm64-apple-tvos9.0.0"

// RUN: env WATCHOS_DEPLOYMENT_TARGET=2 \
// RUN:   %clang -target armv7k-apple-watchos3 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-TWATCHOS-CMD %s
// CHECK-VERSION-TWATCHOS-CMD: "thumbv7k-apple-watchos3.0.0"

// Target with OS version is not overridden by the SDK:

// RUN: %clang -target armv7-apple-ios9 -Wno-missing-sysroot -isysroot SDKs/iPhoneOS11.0.sdk -c -### %s 2>&1 | \
// RUN: FileCheck --check-prefix=CHECK-VERSION-TIOS-SDK %s
// CHECK-VERSION-TIOS-SDK: thumbv7-apple-ios9

// RUN: %clang -target armv7k-apple-watchos4 -Wno-missing-sysroot -isysroot SDKs/WatchOS3.0.sdk -c -### %s 2>&1 | \
// RUN: FileCheck --check-prefix=CHECK-VERSION-TWATCHOS-SDK %s
// CHECK-VERSION-TWATCHOS-SDK: thumbv7k-apple-watchos4

// RUN: %clang -target armv7-apple-tvos9 -Wno-missing-sysroot -isysroot SDKs/AppleTVOS11.0.sdk -c -### %s 2>&1 | \
// RUN: FileCheck --check-prefix=CHECK-VERSION-TTVOS-SDK %s
// CHECK-VERSION-TTVOS-SDK: thumbv7-apple-tvos9

// Target with OS version is not overridden by arch:

// RUN: %clang -target uknown-apple-macos10.11.2 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-TIGNORE-ARCH1 %s
// CHECK-VERSION-TIGNORE-ARCH1: "unknown-apple-macosx10.11.2"

// Target can be used to specify the environment:

// RUN: %clang -target x86_64-apple-ios11-simulator -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-TENV-SIM1 %s
// CHECK-VERSION-TENV-SIM1: "x86_64-apple-ios11.0.0-simulator"

// RUN: %clang -target armv7k-apple-ios10.1-simulator -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-TENV-SIM2 %s
// CHECK-VERSION-TENV-SIM2: "thumbv7k-apple-ios10.1.0-simulator"


// RUN: %clang -target x86_64-apple-macos11 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-MACOS11 %s
// RUN: %clang -target x86_64-apple-darwin20 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-MACOS11 %s
// RUN: %clang -target x86_64-apple-darwin -mmacos-version-min=11 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-MACOS11 %s

// CHECK-MACOS11: "x86_64-apple-macosx11.0.0"

// RUN: %clang -target arm64-apple-macos999 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-MACOS999 %s

// CHECK-MACOS999: "arm64-apple-macosx999.0.0"

// RUN: %clang -target arm64-apple-watchos99 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-WATCHOS99 %s

// CHECK-WATCHOS99: "arm64-apple-watchos99.0.0"

// RUN: not %clang -target arm64-apple-ios999999 -c %s 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-IOS999999 %s

// CHECK-IOS999999: error: invalid version number in '-target arm64-apple-ios999999'
