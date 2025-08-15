/// This test validates that the various ways to assign an invalid deployment version are captured and detected.
// REQUIRES: system-darwin && native

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: env SDKROOT=%t/iPhoneOS21.0.sdk not %clang -m64 -c -### %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SDKROOT

// RUN: not %clang -isysroot %t/iPhoneOS21.0.sdk -m64 -c -### %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SYSROOT

// RUN: not %clang -target arm64-apple-ios21 -c -### %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=TARGET

// RUN: not %clang -mtargetos=ios21 -arch arm64 -c -### %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MTARGET

// RUN: env IPHONEOS_DEPLOYMENT_TARGET=21.0 not %clang -arch arm64 -c -### %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DEPLOY_VAR

// SDKROOT:    error: invalid version number '21.0' inferred from '{{.*}}.sdk'
// SYSROOT:    error: invalid version number '21.0' inferred from '{{.*}}.sdk'
// TARGET:     error: invalid version number in '-target arm64-apple-ios21'
// MTARGET:    error: invalid version number in '-mtargetos=ios21'
// DEPLOY_VAR: error: invalid version number in 'IPHONEOS_DEPLOYMENT_TARGET=21.0'

//--- iPhoneOS21.0.sdk/SDKSettings.json
{"Version":"21.0", "MaximumDeploymentTarget": "21.0.99"}
