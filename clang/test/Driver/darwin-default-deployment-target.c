// Ensure that the deployment target inferred from the SDK when none is
// specified on the command line uses "DefaultDeploymentTarget" rather than
// "Version" when the SDK specifies both and they differ.
// REQUIRES: system-darwin && native

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang -target arm64-apple-darwin -isysroot %t/iPhoneOS18.0.sdk -c -### %s 2>&1 \
// RUN:   | FileCheck %s

// CHECK: "-triple" "arm64-apple-ios17.0.0"
// CHECK-SAME: -target-sdk-version=18.0

// An explicit deployment target on the command line overrides the SDK's
// "DefaultDeploymentTarget".
// RUN: %clang -target arm64-apple-darwin -isysroot %t/iPhoneOS18.0.sdk -miphoneos-version-min=12.0 -c -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=OVERRIDE %s

// OVERRIDE: "-triple" "arm64-apple-ios12.0.0"
// OVERRIDE-SAME: -target-sdk-version=18.0

// When "DefaultDeploymentTarget" is missing, the whole SDKSettings.json is
// treated as unusable and the version is instead inferred from the SDK
// path.
// RUN: %clang -target arm64-apple-darwin -isysroot %t/iPhoneOS18.3.sdk -c -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NO-DEFAULT %s

// NO-DEFAULT: warning: SDK settings were ignored as 'SDKSettings.json' could not be parsed
// NO-DEFAULT: "-triple" "arm64-apple-ios18.3.0"
// NO-DEFAULT-SAME: -target-sdk-version=18.3

//--- iPhoneOS18.0.sdk/SDKSettings.json
{
  "CanonicalName": "iphoneos18.0",
  "Version": "18.0",
  "IsBaseSDK": "YES",
  "DisplayName": "iOS 18.0",
  "MinimalDisplayName": "18.0",
  "SupportedTargets": {
    "iphoneos": {
      "PlatformFamilyName": "iOS",
      "PlatformFamilyDisplayName": "iOS",
      "Archs": ["arm64e", "arm64"], "LLVMTargetTripleVendor": "apple", "LLVMTargetTripleSys": "ios", "LLVMTargetTripleEnvironment": "",
      "BuildVersionPlatformID": "2",
      "ClangRuntimeLibraryPlatformName": "ios",
      "SystemPrefix": "",
      "DefaultDeploymentTarget": "17.0",
      "RecommendedDeploymentTarget": "15.0",
      "MinimumDeploymentTarget": "12.0", "MaximumDeploymentTarget": "18.0.99",
      "ValidDeploymentTargets": ["12.0", "12.1", "12.2", "12.3", "12.4", "13.0", "13.1", "13.2", "13.3", "13.4", "13.5", "13.6", "14.0", "14.1", "14.2", "14.3", "14.4", "14.5", "14.6", "14.7", "15.0", "15.1", "15.2", "15.3", "15.4", "15.5", "15.6", "16.0", "16.1", "16.2", "16.3", "16.4", "16.5", "16.6", "17.0", "17.1", "17.2", "17.3", "17.4", "17.5", "17.6", "18.0"]
    }
  },
  "DefaultDeploymentTarget": "17.0",
  "MaximumDeploymentTarget": "18.0.99",
  "Comments": [
    "Modified version of the iOS SDK from Xcode 18.0 to have \"DefaultDeploymentTarget\" differ from \"Version\"."
  ]
}

//--- iPhoneOS18.3.sdk/SDKSettings.json
{
  "CanonicalName": "iphoneos18.6",
  "Version": "18.6",
  "IsBaseSDK": "YES",
  "DisplayName": "iOS 18.6",
  "MinimalDisplayName": "18.6",
  "SupportedTargets": {
    "iphoneos": {
      "PlatformFamilyName": "iOS",
      "PlatformFamilyDisplayName": "iOS",
      "Archs": ["arm64e", "arm64"], "LLVMTargetTripleVendor": "apple", "LLVMTargetTripleSys": "ios", "LLVMTargetTripleEnvironment": "",
      "BuildVersionPlatformID": "2",
      "ClangRuntimeLibraryPlatformName": "ios",
      "SystemPrefix": "",
      "RecommendedDeploymentTarget": "15.0",
      "MinimumDeploymentTarget": "12.0", "MaximumDeploymentTarget": "18.6.99",
      "ValidDeploymentTargets": ["12.0", "12.1", "12.2", "12.3", "12.4", "13.0", "13.1", "13.2", "13.3", "13.4", "13.5", "13.6", "14.0", "14.1", "14.2", "14.3", "14.4", "14.5", "14.6", "14.7", "15.0", "15.1", "15.2", "15.3", "15.4", "15.5", "15.6", "16.0", "16.1", "16.2", "16.3", "16.4", "16.5", "16.6", "17.0", "17.1", "17.2", "17.3", "17.4", "17.5", "17.6", "18.0", "18.1", "18.2", "18.3", "18.4", "18.5", "18.6"]
    }
  },
  "MaximumDeploymentTarget": "18.6.99",
  "Comments": [
    "Modified version of the iOS SDK from Xcode 18.6 with \"DefaultDeploymentTarget\" removed to test the fallback path used when it's missing. The SDK's folder name deliberately differs from \"Version\" to show the SDK path, not \"Version\", is used for the fallback."
  ]
}
