// RUN: %clang -target x86_64-apple-macosx10.13 -isysroot %S/Inputs/MacOSX10.14.sdk -c -### %s 2>&1 \
// RUN:   | FileCheck %s
// RUN: env SDKROOT=%S/Inputs/MacOSX10.14.sdk %clang -target x86_64-apple-macosx10.13 -c -### %s 2>&1 \
// RUN:   | FileCheck %s
//
// RUN: rm -rf %t/SDKs/MacOSX10.10.sdk
// RUN: mkdir -p %t/SDKs/MacOSX10.10.sdk
// RUN: %clang -m64 -isysroot %t/SDKs/MacOSX10.10.sdk -c -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=INFER_SDK_VERSION %s
// RUN: sed -e 's/10\.14/10\.8/g' %S/Inputs/MacOSX10.14.sdk/SDKSettings.json > %t/SDKs/MacOSX10.10.sdk/SDKSettings.json
// RUN: %clang -m64 -isysroot %t/SDKs/MacOSX10.10.sdk -c -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=INFER_DEPLOYMENT_TARGET_VERSION %s
// REQUIRES: system-darwin && native
//
// RUN: rm -rf %t/SDKs/MacOSX10.14.sdk
// RUN: mkdir -p %t/SDKs/MacOSX10.14.sdk
// RUN: %clang -target x86_64-apple-macosx10.13 -isysroot %t/SDKs/MacOSX10.14.sdk -c -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NO_VERSION %s
//
// RUN: rm -rf %t/SDKs/MacOSX10.14.sdk
// RUN: mkdir -p %t/SDKs/MacOSX10.14.sdk
// RUN: echo '{broken json' > %t/SDKs/MacOSX10.14.sdk/SDKSettings.json
// RUN: %clang -target x86_64-apple-macosx10.13 -isysroot %t/SDKs/MacOSX10.14.sdk -c -### %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO_VERSION,ERROR %s
//
// RUN: rm -rf %t/SDKs/MacOSX10.14.sdk
// RUN: mkdir -p %t/SDKs/MacOSX10.14.sdk
// RUN: echo '{"Version":1}' > %t/SDKs/MacOSX10.14.sdk/SDKSettings.json
// RUN: %clang -target x86_64-apple-macosx10.13 -isysroot %t/SDKs/MacOSX10.14.sdk -c -### %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO_VERSION,ERROR %s

// CHECK: -target-sdk-version=10.14
// INFER_SDK_VERSION: "-triple" "{{arm64|x86_64}}-apple-macosx10.10.0"
// INFER_SDK_VERSION-SAME: -target-sdk-version=10.10
// INFER_DEPLOYMENT_TARGET_VERSION: "-triple" "{{arm64|x86_64}}-apple-macosx10.8.0"
// NO_VERSION-NOT: target-sdk-version
// ERROR: warning: SDK settings were ignored as 'SDKSettings.json' could not be parsed
