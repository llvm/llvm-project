// Ensure that the deployment target inferred from the SDK when none is
// specified on the command line uses "DefaultDeploymentTarget" rather than
// "Version" when the SDK specifies both and they differ.

// RUN: rm -rf %t/SDKs/iPhoneOS18.0.sdk
// RUN: mkdir -p %t/SDKs/iPhoneOS18.0.sdk
// RUN: echo '{"CanonicalName": "iphoneos18.0", "Version": "18.0", "DefaultDeploymentTarget": "17.0", "MaximumDeploymentTarget": "18.0.99"}' \
// RUN:   > %t/SDKs/iPhoneOS18.0.sdk/SDKSettings.json
// RUN: %clang -target arm64-apple-darwin -isysroot %t/SDKs/iPhoneOS18.0.sdk -c -### %s 2>&1 \
// RUN:   | FileCheck %s

// CHECK: "-triple" "arm64-apple-ios17.0.0"
// CHECK-SAME: -target-sdk-version=18.0

// An explicit deployment target on the command line overrides the SDK's
// "DefaultDeploymentTarget".
// RUN: %clang -target arm64-apple-darwin -isysroot %t/SDKs/iPhoneOS18.0.sdk -miphoneos-version-min=12.0 -c -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=OVERRIDE %s

// OVERRIDE: "-triple" "arm64-apple-ios12.0.0"
// OVERRIDE-SAME: -target-sdk-version=18.0
