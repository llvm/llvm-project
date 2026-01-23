// This test checks that the module cache gets invalidated when the
// SDKSettings.json file changes. This prevents "file changed during build"
// errors when the TU does get rescanned and recompiled.

// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- MacOSX11.0.sdk/SDKSettings.json
{
  "CanonicalName": "macosx11.0",
  "Version": "11.0",
  "IsBaseSDK": "YES",
  "DisplayName": "macOS 11.0",
  "MinimalDisplayName": "11.0",
  "SupportedTargets": {
    "macosx": {
      "PlatformFamilyName": "macOS",
      "Archs": ["x86_64", "x86_64h", "arm64", "arm64e"], "LLVMTargetTripleVendor": "apple", "LLVMTargetTripleSys": "macosx", "LLVMTargetTripleEnvironment": "",
      "BuildVersionPlatformID": "1",
      "DefaultDeploymentTarget": "11.0",
      "MinimumDeploymentTarget": "10.9", "MaximumDeploymentTarget": "11.0.99",
      "ValidDeploymentTargets": ["10.9", "10.10", "10.11", "10.12", "10.13", "10.14", "10.15", "11.0"]
    },
    "iosmac": {
      "Archs": ["x86_64", "x86_64h", "arm64", "arm64e"], "LLVMTargetTripleVendor": "apple", "LLVMTargetTripleSys": "ios", "LLVMTargetTripleEnvironment": "macabi",
      "BuildVersionPlatformID": "6",
      "DefaultDeploymentTarget": "14.2",
      "MinimumDeploymentTarget": "13.1", "MaximumDeploymentTarget": "14.2.99",
      "ValidDeploymentTargets": ["13.1", "13.2", "13.3", "13.3.1", "13.4", "13.5", "14.2"]
    }
  },
  "VersionMap": {
    "macOS_iOSMac": {"10.15": "13.1", "10.15.1": "13.2", "10.15.2": "13.3", "10.15.3": "13.3.1", "10.15.4": "13.4", "10.15.5": "13.5", "11.0": "14.2"},
    "iOSMac_macOS": {"13.1": "10.15", "13.2": "10.15.1", "13.3": "10.15.2", "13.3.1": "10.15.3", "13.4": "10.15.4", "13.5": "10.15.5", "14.2": "11.0"}
  },
  "DefaultDeploymentTarget": "11.0",
  "MaximumDeploymentTarget": "11.0.99"
}

//--- module.modulemap
module M { header "M.h" }
//--- M.h
//--- tu.c
#include "M.h"

// RUN: clang-scan-deps -format experimental-include-tree-full -cas-path %t/cas -o %t/deps_clean.json \
// RUN:   -- %clang -target x86_64-apple-macos11 -isysroot %t/MacOSX11.0.sdk \
// RUN:      -c %t/tu.c -o %t/tu.o -fmodules -fmodules-cache-path=%t/cache

// RUN: sleep 1
// RUN: echo " " >> %t/MacOSX11.0.sdk/SDKSettings.json
// RUN: echo " " >> %t/tu.c

// RUN: clang-scan-deps -format experimental-include-tree-full -cas-path %t/cas -o %t/deps_incremental.json \
// RUN:   -- %clang -target x86_64-apple-macos11 -isysroot %t/MacOSX11.0.sdk \
// RUN:      -c %t/tu.c -o %t/tu.o -fmodules -fmodules-cache-path=%t/cache

// RUN: %deps-to-rsp %t/deps_incremental.json --module-name M > %t/M.rsp
// RUN: %deps-to-rsp %t/deps_incremental.json --tu-index 0 > %t/tu.rsp
// RUN: %clang @%t/M.rsp
// RUN: %clang @%t/tu.rsp
