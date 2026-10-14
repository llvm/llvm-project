// This test checks that the module cache gets invalidated when the SDKSettings.json file changes.

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- AppleTVOS15.0.sdk/SDKSettings-old.json
{
  "CanonicalName": "appletvos15.0",
  "Version": "15.0",
  "IsBaseSDK": "YES",
  "DisplayName": "tvOS 15.0",
  "MinimalDisplayName": "15.0",
  "SupportedTargets": {
    "appletvos": {
      "PlatformFamilyName": "tvOS",
      "Archs": ["arm64e", "arm64"], "LLVMTargetTripleVendor": "apple", "LLVMTargetTripleSys": "tvos", "LLVMTargetTripleEnvironment": "",
      "BuildVersionPlatformID": "3",
      "DefaultDeploymentTarget": "15.0",
      "MinimumDeploymentTarget": "9.0", "MaximumDeploymentTarget": "15.0.99",
      "ValidDeploymentTargets": ["9.0", "9.1", "9.2", "10.0", "10.1", "10.2", "11.0", "11.1", "11.2", "11.3", "11.4", "12.0", "12.1", "12.2", "12.3", "12.4", "13.0", "13.1", "13.2", "13.3", "13.4", "14.0", "14.1", "14.2", "14.3", "14.4", "14.5", "14.6", "14.7", "15.0"]
    }
  },
  "DefaultDeploymentTarget": "15.0",
  "MaximumDeploymentTarget": "15.0.99",
  "Comments": [
    "Modified version of the tvOS SDK from Xcode 13.0 to remove VersionMap."
  ]
}
//--- AppleTVOS15.0.sdk/SDKSettings-new.json
{
  "CanonicalName": "appletvos15.0",
  "Version": "15.0",
  "IsBaseSDK": "YES",
  "DisplayName": "tvOS 15.0",
  "MinimalDisplayName": "15.0",
  "SupportedTargets": {
    "appletvos": {
      "PlatformFamilyName": "tvOS",
      "Archs": ["arm64e", "arm64"], "LLVMTargetTripleVendor": "apple", "LLVMTargetTripleSys": "tvos", "LLVMTargetTripleEnvironment": "",
      "BuildVersionPlatformID": "3",
      "DefaultDeploymentTarget": "15.0",
      "MinimumDeploymentTarget": "9.0", "MaximumDeploymentTarget": "15.0.99",
      "ValidDeploymentTargets": ["9.0", "9.1", "9.2", "10.0", "10.1", "10.2", "11.0", "11.1", "11.2", "11.3", "11.4", "12.0", "12.1", "12.2", "12.3", "12.4", "13.0", "13.1", "13.2", "13.3", "13.4", "14.0", "14.1", "14.2", "14.3", "14.4", "14.5", "14.6", "14.7", "15.0"]
    }
  },
  "VersionMap": {
    "tvOS_iOS": {"9.0": "9.0",               "9.1": "9.2", "9.2": "9.3", "10.0": "10.0",                 "10.1": "10.2", "10.2": "10.3",                   "11.0": "11.0", "11.1": "11.1", "11.2": "11.2", "11.3": "11.3", "11.4": "11.4", "12.0": "12.0", "12.1": "12.1", "12.2": "12.2", "12.4": "12.4", "13.0": "13.0",                 "13.2": "13.2", "13.4": "13.4",                                                 "14.0": "14.0",                 "14.2": "14.2", "14.3": "14.3",                 "14.5": "14.5", "15.0": "15.0"},
    "iOS_tvOS": {"9.0": "9.0", "9.1": "9.0", "9.2": "9.1", "9.3": "9.2", "10.0": "10.0", "10.1": "10.0", "10.2": "10.1", "10.3": "10.2", "10.3.1": "10.2", "11.0": "11.0", "11.1": "11.1", "11.2": "11.2", "11.3": "11.3", "11.4": "11.4", "12.0": "12.0", "12.1": "12.1", "12.2": "12.2", "12.4": "12.4", "13.0": "13.0", "13.1": "13.0", "13.2": "13.2", "13.4": "13.4", "13.5": "13.4", "13.6": "13.4", "13.7": "13.4", "14.0": "14.0", "14.1": "14.0", "14.2": "14.2", "14.3": "14.3", "14.4": "14.3", "14.5": "14.5", "15.0": "15.0"}
  },
  "DefaultDeploymentTarget": "15.0",
  "MaximumDeploymentTarget": "15.0.99",
  "Comments": [
    "Unmodified version of the tvOS SDK from Xcode 13.0."
  ]
}
//--- module.modulemap
module M { header "M.h" }
//--- M.h
void foo(void) __attribute__((availability(iOS, obsoleted = 10.3)));
void test() { foo(); }

//--- tu.m
#include "M.h"

// Compiling for tvOS 10.2 without "VersionMap" should succeed, since by default iOS 10.3 gets mapped to tvOS 10.3,
// and \c foo is therefore **not** deprecated.
// RUN: cp %t/AppleTVOS15.0.sdk/SDKSettings-old.json %t/AppleTVOS15.0.sdk/SDKSettings.json
// RUN: %clang -target x86_64-apple-tvos10.2 -isysroot %t/AppleTVOS15.0.sdk \
// RUN:   -fsyntax-only %t/tu.m -o %t/tu.o -fmodules -Xclang -fdisable-module-hash -fmodules-cache-path=%t/cache

// Compiling for tvOS 10.2 with "VersionMap" saying it maps to iOS 10.3 should fail, since \c foo is now deprecated.
// RUN: sleep 1
// RUN: cp %t/AppleTVOS15.0.sdk/SDKSettings-new.json %t/AppleTVOS15.0.sdk/SDKSettings.json
// RUN: not %clang -target x86_64-apple-tvos10.2 -isysroot %t/AppleTVOS15.0.sdk \
// RUN:   -fsyntax-only %t/tu.m -o %t/tu.o -fmodules -Xclang -fdisable-module-hash -fmodules-cache-path=%t/cache 2>&1 \
// RUN:     | FileCheck %s
// CHECK: M.h:2:15: error: 'foo' is unavailable: obsoleted in tvOS 10.2
// CHECK: M.h:1:6: note: 'foo' has been explicitly marked unavailable here
// CHECK: tu.m:1:10: fatal error: could not build module 'M'
