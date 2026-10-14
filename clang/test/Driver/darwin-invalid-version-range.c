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
{
  "CanonicalName": "iphoneos21.0",
  "Version": "21.0",
  "IsBaseSDK": "YES",
  "DisplayName": "iOS 21.0",
  "MinimalDisplayName": "21.0",
  "SupportedTargets": {
    "iphoneos": {
      "PlatformFamilyName": "iOS",
      "PlatformFamilyDisplayName": "iOS",
      "Archs": ["arm64e", "arm64"], "LLVMTargetTripleVendor": "apple", "LLVMTargetTripleSys": "ios", "LLVMTargetTripleEnvironment": "",
      "BuildVersionPlatformID": "2",
      "ClangRuntimeLibraryPlatformName": "ios",
      "SystemPrefix": "",
      "DefaultDeploymentTarget": "21.0",
      "RecommendedDeploymentTarget": "15.0",
      "MinimumDeploymentTarget": "12.0", "MaximumDeploymentTarget": "21.0.99",
      "ValidDeploymentTargets": ["12.0", "12.1", "12.2", "12.3", "12.4", "13.0", "13.1", "13.2", "13.3", "13.4", "13.5", "13.6", "14.0", "14.1", "14.2", "14.3", "14.4", "14.5", "14.6", "14.7", "15.0", "15.1", "15.2", "15.3", "15.4", "15.5", "15.6", "16.0", "16.1", "16.2", "16.3", "16.4", "16.5", "16.6", "17.0", "17.1", "17.2", "17.3", "17.4", "17.5", "17.6", "18.0", "18.1", "18.2", "18.3", "18.4", "18.5", "18.6", "21.0"]
    }
  },
  "DefaultDeploymentTarget": "21.0",
  "MaximumDeploymentTarget": "21.0.99",
  "Comments": [
    "Modified version of the iOS SDK from Xcode 21.0 to have an invalid version."
  ]
}
